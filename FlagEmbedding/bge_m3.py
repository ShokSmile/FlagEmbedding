from typing import cast, List, Union, Tuple, Optional, Dict
import numpy as np
from collections import defaultdict
import torch
from tqdm import tqdm
import datasets
from transformers import PreTrainedTokenizerFast, BatchEncoding, DataCollatorWithPadding, XLMRobertaForMaskedLM, is_torch_npu_available
from torch.utils.data import DataLoader
from functools import partial
from FlagEmbedding.BGE_M3 import BGEM3ForInference


def _transform_func(examples: Dict[str, List],
                    tokenizer: PreTrainedTokenizerFast,
                    max_length: int = 8192,
                    ) -> BatchEncoding:
    inputs = tokenizer(examples['text'],
                       max_length=max_length,
                       padding=True,
                       return_token_type_ids=False,
                       truncation=True,
                       return_tensors='pt')
    return inputs


class BGEM3FlagModel:
    def __init__(
            self,
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            use_fp16: bool = True,
            device: str = None
    ) -> None:

        self.model = BGEM3ForInference(
            model_name=model_name_or_path,
            normlized=normalize_embeddings,
            sentence_pooling_method=pooling_method,
        )

        self.tokenizer = self.model.tokenizer
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif is_torch_npu_available():
                self.device = torch.device("npu")
            else:
                self.device = torch.device("cpu")
                use_fp16 = False
        if use_fp16: self.model.half()
        self.model = self.model.to(self.device)

        if device is None:
            self.num_gpus = torch.cuda.device_count()
            if self.num_gpus > 1:
                print(f"----------using {self.num_gpus}*GPUs----------")
                self.model.model = torch.nn.DataParallel(self.model.model)
        else:
            self.num_gpus = 1

        self.model.eval()

    def convert_id_to_token(self, lexical_weights: List[Dict]):
        if isinstance(lexical_weights, dict):
            lexical_weights = [lexical_weights]
        new_lexical_weights = []
        for item in lexical_weights:
            new_item = {}
            for id, weight in item.items():
                token = self.tokenizer.decode([int(id)])
                new_item[token] = weight
            new_lexical_weights.append(new_item)

        if len(new_lexical_weights) == 1:
            new_lexical_weights = new_lexical_weights[0]
        return new_lexical_weights

    def compute_lexical_matching_score(self, lexical_weights_1: Dict, lexical_weights_2: Dict):
        scores = 0
        for token, weight in lexical_weights_1.items():
            if token in lexical_weights_2:
                scores += weight * lexical_weights_2[token]
        return scores

    def colbert_score(self, reps, p_reps):
        reps, p_reps = torch.from_numpy(reps), torch.from_numpy(p_reps)
        token_scores = torch.einsum('in,jn->ij', reps, p_reps)
        scores, _ = token_scores.max(-1)
        scores = torch.sum(scores) / reps.size(0)
        return scores


    @torch.no_grad()
    def encode(self,
               sentences: Union[List[str], str],
               batch_size: int = 12,
               max_length: int = 8192,
               return_dense: bool = True,
               return_sparse: bool = False,
               return_colbert_vecs: bool = False) -> Dict:
        
        # print(self.num_gpus)

        if self.num_gpus > 1:
            batch_size *= self.num_gpus
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        def _process_token_weights(token_weights: np.ndarray, input_ids: list):
            # conver to dict
            result = defaultdict(int)
            unused_tokens = set([self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,
                                 self.tokenizer.unk_token_id])
            # token_weights = np.ceil(token_weights * 100)
            for w, idx in zip(token_weights, input_ids):
                if idx not in unused_tokens and w > 0:
                    idx = str(idx)
                    # w = int(w)
                    if w > result[idx]:
                        result[idx] = w
            return result

        def _process_colbert_vecs(colbert_vecs: np.ndarray, attention_mask: list):
            # delte the vectors of padding tokens
            tokens_num = np.sum(attention_mask)
            return colbert_vecs[:tokens_num - 1]  # we don't use the embedding of cls, so select tokens_num-1


        all_dense_embeddings, all_lexical_weights, all_colbert_vec = [], [], []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            batch_data = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to(self.device)
            output = self.model(batch_data,
                                return_dense=return_dense,
                                return_sparse=return_sparse,
                                return_colbert=return_colbert_vecs)
            if return_dense:
                all_dense_embeddings.append(output['dense_vecs'].cpu().numpy())

            if return_sparse:
                token_weights = output['sparse_vecs'].squeeze(-1)
                all_lexical_weights.extend(list(map(_process_token_weights, token_weights.cpu().numpy(),
                                                    batch_data['input_ids'].cpu().numpy().tolist())))

            if return_colbert_vecs:
                all_colbert_vec.extend(list(map(_process_colbert_vecs, output['colbert_vecs'].cpu().numpy(),
                                                batch_data['attention_mask'].cpu().numpy())))

        if return_dense:
            all_dense_embeddings = np.concatenate(all_dense_embeddings, axis=0)

        if return_dense:
            if input_was_string:
                all_dense_embeddings = all_dense_embeddings[0]
        else:
            all_dense_embeddings = None

        if return_sparse:
            if input_was_string:
                all_lexical_weights = all_lexical_weights[0]
        else:
            all_lexical_weights = None

        if return_colbert_vecs:
            if input_was_string:
                all_colbert_vec = all_colbert_vec[0]
        else:
            all_colbert_vec = None

        return {"dense_vecs": all_dense_embeddings, "lexical_weights": all_lexical_weights,
                "colbert_vecs": all_colbert_vec}

    @torch.no_grad()
    def compute_score_matrix(self,
                      sentences: List[str],
                      return_embeddings: bool = True,
                      batch_size: int = 16,
                      model_max_length: int = 300,
                      save_tensor: bool = False, 
                      path_to_save: str = "",
                      weights_for_different_modes: List[float] = None) -> Dict[str, List[float]]:
        
        """
    This function encodes input and calculate score for each representation 

    Args:
        sentences (List[str]): That's a list of strings. You have to preprocess your input before calling this function. It means choose format of input (title, description, title+description or another)
        save_tensor (bool): Should we save torch.Tensor with embeddings or not.
        path_to_save (str): path where it's necessary to save model
        model_max_length (int): max input length if it's bigger -> truncation
        weights_for_different_modes (List[float]): weights of sum of all representations
        batch_size (int): batch size
    Returns:
        torch.Tensor: tensor with whole representations depending on return_dense, return_sparse, return_colbert_vecs.
        
    TODO:
    1. Add weighted sum for different modes
    """

        def _tokenize(texts: list, max_length: int):
            return self.tokenizer(
                texts,
                max_length=max_length,
                padding=True,
                return_token_type_ids=False,
                truncation=True,
                return_tensors='pt'
            )

        if self.num_gpus > 0:
            batch_size *= self.num_gpus
        
        if isinstance(sentences, list) and len(sentences) == 0:
            return []

        # sim_matrix = torch.zeros(3, len(sentences), len(sentences), dtype=torch.float)
        dense_emb = torch.zeros(len(sentences), 1024)
        
        
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Compute scores and encode sentences..."):
            if start_index + batch_size >= len(sentences):
                queries_batch = sentences[start_index:]
            else:
                queries_batch = sentences[start_index:start_index + batch_size]

            queries_inputs = _tokenize(queries_batch, max_length=model_max_length).to(self.device)
            
            queries_output = self.model(queries_inputs, return_dense=True,
                                        return_sparse=False,
                                        return_colbert=False,
                                        return_sparse_embedding=False)
            
            # for the moment because of the sizes of emb matrices -> we hate make it only for dense representations
            # dense_vecs, sparse_vecs, colbert_vecs = queries_output['dense_vecs'], queries_output['sparse_vecs'], \
            # queries_output['colbert_vecs']
            
            dense_vecs = queries_output['dense_vecs']
            
            if start_index + batch_size >= len(sentences):
                dense_emb[start_index:, :] = dense_vecs
            else:
                dense_emb[start_index: start_index + batch_size, :] = dense_vecs
            
            # print(f"""
            #       ------------------------
            #       Dense representation
            #       type: {type(dense_vecs)}
            #       dense vec : {dense_vecs[0]}
            #       shape: {dense_vecs.shape}
            #       ------------------------
                  
                  
            #       ------------------------
            #       Sparse representation
            #       type: {type(sparse_vecs)}
            #       type_1: {sparse_vecs[0]}
            #       shape: {sparse_vecs.shape}
            #       ------------------------
                  
            #        ------------------------
            #       Colbert representation
            #       type: {type(colbert_vecs)}
            #       type_1: {colbert_vecs[0]}
            #       shape: {colbert_vecs.shape}
            #       ------------------------
            #       """)

            #TODO: add tensors where we'll save our representations

            # dense_scores = self.model.dense_score(dense_vecs, dense_vecs)
            # sparse_scores = self.model.sparse_score(sparse_vecs, sparse_vecs)
            # colbert_scores = self.model.colbert_score(colbert_vecs, colbert_vecs,
            #                                           q_mask=queries_inputs['attention_mask'])
            
            # print(dense_scores)
            # print(sparse_scores)
            # print(colbert_scores)
            
            # # TODO: add sim matrix
        dense_emb.cpu()
        sim_matrix = self.model.compute_similarity(dense_emb, dense_emb).half()
        if return_embeddings:
            return dense_emb, sim_matrix
        else:
            return sim_matrix
    
if __name__ == "__main__":
    import pandas as pd
    import json
    
    
    # # read initial csv
    # data = pd.read_csv("/Users/sanek_tarasov/Nukema/data/contrat_attribue.csv")

    # # target == nb offre recu
    # target = data['nb_offre_recu']

    # # indexes of initial data (not NaN)
    # indexes = data[(data['source_contrat'] == 'BOAMP') & (~pd.isna(data['nb_offre_recu']))]['nb_offre_recu'].index

    # # read json file with pretokenized data: title, description, title+description
    # with open("/Users/sanek_tarasov/Nukema/data/tokenized_data_with_mix.json") as f:
    #     data_dict = json.load(f)

    # # Choosing samples from pre-tokenized data dict which are not NaN (using indexes)
    # data_dict = [data_dict[i] for i in tqdm(indexes, desc="Choosing samples from pre-tokenized data dict")]
    # # sometimes we have empty descriptions (NaN) -> change it to empty str
    # for i in range(len(data_dict)):
    #     if pd.isna(data_dict[i]['description']):
    #         data_dict[i]['description'] = ""
    #     if pd.isna(data_dict[i]['title']):
    #         data_dict[i]['title'] = "" 
    
    
    test = ["Hello world",
            "Bye World", 
            "How are you going",
            "ha ah dkkm dksmdks dkdne"]
    
    
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, device="cpu")
    model.compute_score_matrix(test)
    


