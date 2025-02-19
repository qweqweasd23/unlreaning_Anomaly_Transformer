import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment
from tqdm import tqdm
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tempfile
import dask.array as da
import tempfile
from scipy.ndimage import uniform_filter
from scipy.interpolate import interp1d
#from pytdigest import TDigest

def my_kl_loss(p, q):
#    # p와 q는 연속형 확률 분포를 나타내는 텐서입니다.
    p = torch.clamp(p, min=1e-10)  # p가 0이 되는 것을 방지
    q = torch.clamp(q, min=1e-10)  # q가 0이 되는 것을 방지
    res = p * (torch.log(p) - torch.log(q))
    return torch.mean(torch.sum(res, dim=-1))
#
def bhattacharyya_distance(p, q, eps=1e-10):
    """
    p, q: 마지막 차원이 확률분포(합이 1이 되도록 정규화된 텐서)
    eps: 수치 안정성을 위한 작은 값
    반환: 각 배치 혹은 샘플별 Bhattacharyya Distance (스칼라 값)
    """
    # 각 요소의 제곱근을 취한 후, 마지막 차원에서 합산
    bc = torch.sum(torch.sqrt(p * q + eps), dim=-1)
    # Bhattacharyya Distance 계산
    bd = -torch.log(bc + eps)
    return bd

def reconstruct_from_windows_memmap(prediction, total_row, win_size, output_path):
    """
    슬라이딩 윈도우 기반 prediction 데이터를 복구(reconstruct)합니다.
    중복된 위치에서 앞 또는 뒤 값이 1인 경우 1로 설정하며, 결과를 memmap으로 저장합니다.

    Args:
        prediction (np.ndarray): Shape (len, win, feat)의 prediction 데이터.
        win_size (int): 슬라이딩 윈도우 크기.
        output_path (str): 복구된 데이터를 저장할 memmap 파일 경로.

    Returns:
        np.memmap: 복구된 prediction 데이터. Shape (len + win_size - 1, feat).
    """
    len_full, win, feat = prediction.shape
    reconstructed_shape = (len_full + win_size - 1, feat)

    # Memmap 파일 생성
    reconstructed_memmap = np.memmap(output_path, dtype='float32', mode='w+', shape=reconstructed_shape)
    count_memmap = np.memmap(output_path + "_count", dtype='int32', mode='w+', shape=reconstructed_shape)

    # 초기화
    reconstructed_memmap[:] = 0
    count_memmap[:] = 0

    # 슬라이딩 윈도우를 통해 복구   여가기에 tqdm 추가 , 지금 매우 병목인듯 
    for t in tqdm(range(len_full),desc='reconsturct'):
        start_idx = t
        end_idx = t + win_size
        if end_idx > len_full + win_size - 1:
            break

        reconstructed_memmap[start_idx:end_idx] += prediction[t, :, :]
        count_memmap[start_idx:end_idx] += 1

    # 중복된 위치에서 하나라도 1인 경우 1로 설정
    reconstructed_memmap[:] = np.where(reconstructed_memmap > 0, 1, 0)

    # Memmap 플러시 및 반환
    reconstructed_memmap.flush()
    #count_memmap.flush()

    # Count memmap 파일 삭제 (필요 없으므로)
    del count_memmap

    return reconstructed_memmap



def aggregate_anomalies(
    prediction, aggregate_unit, min_length
):
    """
    Aggregate anomalies based on prediction, aggregation unit, and minimum anomaly length.
    
    Args:
        prediction (np.ndarray): Binary anomaly predictions of shape (data_len, feat_num).
        aggregate_unit (int): Number of time steps to aggregate into one result.
        min_length (int): Minimum length of continuous anomalies to be considered as anomaly.
        
    Returns:
        np.ndarray: Aggregated anomaly results of shape (num_aggregates, feat_num).
    """
    # Extract data shape
    data_len, feat_num = prediction.shape

    # Calculate the number of aggregate units
    num_aggregates = data_len // aggregate_unit

    # Initialize the aggregated results array
    aggregated_results = np.zeros((num_aggregates, feat_num), dtype=int)

    # Process each aggregation unit
    for unit_idx in tqdm(range(num_aggregates),desc='aggregate'):
        start_idx = unit_idx * aggregate_unit
        end_idx = start_idx + aggregate_unit

        # For each feature column, check for anomalies in the current aggregation unit
        for col in range(feat_num):
            consecutive_anomalies = 0
            is_anomalous = False

            for i in range(start_idx, end_idx):
                if prediction[i, col] == 1:
                    consecutive_anomalies += 1
                    if consecutive_anomalies >= min_length:
                        is_anomalous = True
                        break
                else:
                    consecutive_anomalies = 0

            # Set the aggregated value for this feature column
            aggregated_results[unit_idx, col] = int(is_anomalous)

    return aggregated_results

def process_batches(test_scores_path, prediction_path, threshold_path,
                    num_batches, batch_size, win_size, target_columns,
                    zscore_threshold,total_samples):
    """
    배치 단위로 threshold와 prediction을 계산하여 메모리 사용량을 줄이는 함수.

    Args:
        test_scores_path (str): Test scores memmap 경로.
        prediction_path (str): Prediction 결과 저장 경로.
        threshold_path (str): Threshold 결과 저장 경로.
        num_batches (int): 총 배치 수.
        batch_size (int): 배치 크기.
        win_size (int): 슬라이딩 윈도우 크기.
        target_columns (list): Feature 칼럼 리스트.
        zscore_threshold (float): Z-score 임계값.
    """
    # Test scores memmap 로드
    test_scores_memmap = np.memmap(test_scores_path, dtype='float32', mode='r',
                                   shape=(total_samples-(win_size-1), win_size, len(target_columns)))

    # Prediction 및 Threshold 저장을 위한 memmap 생성
    prediction_memmap = np.memmap(prediction_path, dtype='float32', mode='w+',
                                  shape=test_scores_memmap.shape)
    threshold_memmap = np.memmap(threshold_path, dtype='float32', mode='w+',
                                 shape=test_scores_memmap.shape)


    for batch_idx in tqdm(range(num_batches), desc="Processing Batches"):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        # 현재 배치의 데이터 로드
        test_scores_batch = test_scores_memmap[start_idx:end_idx]

        for feat_idx in range(test_scores_batch.shape[2]):  # 각 feature에 대해 처리
            for t in range(test_scores_batch.shape[0]):     # 각 시점에 대해 처리
                if t >= win_size:
                    window_data = test_scores_batch[t-win_size:t, :, feat_idx]
                    mean_window = np.mean(window_data, axis=0)
                    std_window = np.std(window_data, axis=0)

                    # Threshold 계산 및 저장
                    threshold_value = mean_window - zscore_threshold * std_window
                    threshold_memmap[start_idx + t, :, feat_idx] = threshold_value

                    # Prediction 계산 및 저장
                    prediction_memmap[start_idx + t, :, feat_idx] = (
                        test_scores_batch[t, :, feat_idx] < threshold_value).astype(float)
                else:
                    # 초기 구간 처리
                    threshold_memmap[start_idx + t, :, feat_idx] = np.nan
                    prediction_memmap[start_idx + t, :, feat_idx] = 0

    # Memmap 파일 닫기 및 저장 플러시
    
    print(f'thres, pred.shape : {threshold_memmap.shape},{prediction_memmap.shape}')
    threshold_memmap.flush()
    prediction_memmap.flush()
    threshold_memmap._mmap.close()
    del threshold_memmap
    torch.cuda.empty_cache()
    gc.collect()
    
    return prediction_memmap

    
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0, study_name=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.study_name = study_name
    def __call__(self, val_loss, val_loss2, model, path):
        # score 계산 (loss1은 음수 대비)
        score = -val_loss #if val_loss < 0 else -val_loss  # loss1 처리
        score2 = -val_loss2  # loss2 처리

        # 초기화
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        # 비교 로직 (부호에 따라 동작 조정)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #model.cpu() 
        torch.save(model.state_dict(), os.path.join(path, f'{str(self.dataset)}_{str(self.study_name)}_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        # 설정 초기화
        self.__dict__.update(Solver.DEFAULTS, **config)

        # 데이터 로더 설정 (레이블 없이 사용)
        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train', dataset=self.dataset,memmap_TF=self.memmap_TF,total_row=self.total_row,input_c=self.input_c)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val', dataset=self.dataset,memmap_TF=self.memmap_TF,total_row=self.total_row,input_c=self.input_c)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test', dataset=self.dataset,memmap_TF=self.memmap_TF,total_row=self.total_row,input_c=self.input_c)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre', dataset=self.dataset,memmap_TF=self.memmap_TF,total_row=self.total_row,input_c=self.input_c)
        
        # 모델 초기화
        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler1 = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.7, patience=self.scheduler_patience)
        self.scheduler2 = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.7, patience=self.scheduler_patience)

    def build_model(self):
        # Anomaly Transformer 모델 초기화
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        if torch.cuda.is_available():
            self.model.cuda()
    def create_pad_mask(self, masking_cols, batch_size, win_size):
        pad_mask = torch.ones(batch_size, win_size).to(self.device)  # Use actual batch size
        for col in masking_cols:
            pad_mask[:, col] = 0
        return pad_mask
    def vali(self, vali_loader):
        # 검증 단계 (비지도 학습)
        self.model.eval()
        loss_1 = []
        loss_2 = []

        for input_data in vali_loader:  # 레이블 제거
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            # Association discrepancy 계산
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):

                series_loss += (torch.mean(my_kl_loss(series[u], 
                                                      (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)))

                                 +
                                torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)).detach(),

                                                      series[u]))))
                prior_loss += torch.mean(bhattacharyya_distance(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size))))

            series_loss = series_loss/len(prior)
            prior_loss = prior_loss*self.prior_weight /len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())
        print(f'val: rec_loss: {rec_loss.item():.6f}, series_loss: {series_loss.item():.6f}, prior_loss: {prior_loss.item():.6f}')    
        return np.average(loss_1), np.average(loss_2)
        

    def train(self):
        print("======================TRAIN MODE======================")

        train_loss_log=[]
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        
        early_stopping = EarlyStopping(patience=20, verbose=True, dataset_name=self.dataset, study_name=self.study_name)
        #train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss_list = []
            loss_list2=[]

            
            self.model.train()

            for input_data in self.train_loader:  # 레이블 제거
                input = input_data.float().to(self.device)

                self.optimizer.zero_grad()

                output, series, prior, _ = self.model(input)

                # Association discrepancy 계산
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):

                    series_loss += (torch.mean(my_kl_loss(series[u], 
                                                          (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach())) 

                                                          +
                                    torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach(),
                                                          series[u])))

                    prior_loss += torch.mean(bhattacharyya_distance(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size))))
                series_loss = series_loss/len(prior)
                prior_loss = prior_loss*self.prior_weight /len(prior)
                rec_loss = self.criterion(output, input)
                loss_total = rec_loss - self.k * series_loss 
                loss_list.append(loss_total.item())
                loss2 = rec_loss + self.k * prior_loss 
                loss_list2.append(loss2.item())
                # 역전파 및 최적화
                loss_total.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

                iter_count += 1
            #print(f'train: prior len: {len(prior)}')
            print(f"Epoch {epoch + 1}, Loss: {np.average(loss_list)}")
            print(f'Learning rate: {self.optimizer.param_groups[0]["lr"]}')
            print(f'train: rec_loss: {rec_loss.item():.6f}, series_loss: {series_loss.item():.6f}, prior_loss: {prior_loss.item():.6f}')
            #print(f'Train Loss: {rec_loss.item()-self.k*series_loss.item():.6f}, {rec_loss.item()+self.k*prior_loss.item():.6f}')
            vali_loss_1, vali_loss_2 = self.vali(self.vali_loader)
            #print(f"Validation Loss: {vali_loss_1:.6f}, {vali_loss_2:.6f}")
            self.scheduler1.step(np.average(loss_list))
            self.scheduler2.step(np.average(loss_list2))
            early_stopping(vali_loss_1 , vali_loss_2 , self.model, self.model_save_path)
            train_loss_log.append([epoch+1,vali_loss_1,vali_loss_2])
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
        np.save(f"{path}/train_loss_log.npy", train_loss_log)

    def finetune(self):
        print("======================finetune MODE======================")

        finetune_loss_log=[]
        self.model.load_state_dict(
            torch.load(self.pretrained_path,weights_only=True)
        )
        
        early_stopping = EarlyStopping(patience=20, verbose=True, dataset_name=self.dataset, study_name=self.study_name)
        #train_steps = len(self.train_loader)

        pad_mask = None

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss_list = []
            loss_list2=[]
            
            self.model.train()

            for input_data in self.train_loader:  # 레이블 제거
                input = input_data.float().to(self.device)
                #패드 마스킹 생성
                if self.masking_cols is not None:
                    # Inside the forward method
                    batch_size = input.shape[0]  # Get actual batch size from input data
                    win_size = input.shape[1]    # Get sequence length (window size)
                    pad_mask = self.create_pad_mask(self.masking_cols, batch_size, win_size)


                self.optimizer.zero_grad()

                output, series, prior, _ = self.model(input,pad_mask=pad_mask)

                # Association discrepancy 계산
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):

                    series_loss += (torch.mean(my_kl_loss(series[u], 
                                                          (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach())) 

                                                          +
                                    torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach(),
                                                          series[u])))

                    prior_loss += torch.mean(bhattacharyya_distance(series[u].detach(), (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size))))

                series_loss = series_loss/len(prior)
                prior_loss = prior_loss*self.prior_weight /len(prior)
                rec_loss = self.criterion(output, input)
                loss_total = rec_loss - self.k * series_loss 
                loss_list.append(loss_total.item())
                loss2 =rec_loss + self.k * prior_loss
                loss_list2.append(loss2.item())
                # 역전파 및 최적화
                loss_total.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

                iter_count += 1
            #print(f'train: prior len: {len(prior)}')
            print(f"Epoch {epoch + 1}, Loss: {np.average(loss_list)}")
            print(f'Learning rate: {self.optimizer.param_groups[0]["lr"]}')
            print(f'ft: rec_loss: {rec_loss.item():.6f}, series_loss: {series_loss.item():.6f}, prior_loss: {prior_loss.item():.6f}')
            print(f'ft Loss: {rec_loss.item()-self.k*series_loss.item():.6f}, {rec_loss.item()+self.k*prior_loss.item():.6f}')
            vali_loss_1, vali_loss_2 = self.vali(self.vali_loader)
            print(f"ft_val Loss: {vali_loss_1:.6f}, {vali_loss_2:.6f}")
            self.scheduler1.step(np.average(loss_list))
            self.scheduler2.step(np.average(loss_list2))
            early_stopping(vali_loss_1 , vali_loss_2 , self.model, self.model_save_path)
            finetune_loss_log.append([epoch+1,vali_loss_1,vali_loss_2])
            if early_stopping.early_stop:
                print("Early stopping")
                break
        np.save(f"{self.model_save_path}/finetune_loss_log.npy", finetune_loss_log)


    def test(self):
        # 1. 훈련된 모델 로드
        self.model.load_state_dict(
            torch.load(os.path.join(self.model_save_path, f"{self.dataset}_{self.study_name}_finetune_checkpoint.pth"),weights_only=True)
        )
        self.model.eval()
        temperature = 50  # KL-divergence scaling factor

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduction='none')
        
        # 2. 훈련 데이터에서 이상 점수 계산 (통계적 기준 설정)
        target_columns = self.target_columns
        #print(f"Target columns: {target_columns}")
        # 3. 테스트 데이터에서 이상 탐지 및 z-score 기반 threshold 생성
        print("Generating test scores and detecting anomalies using z-score-based thresholding")
        #print(f'input_len : {len(self.test_loader)}')
        num_batches = self.total_row // self.batch_size + (len(self.test_loader) % self.batch_size > 0)
        total_samples = self.total_row #* self.batch_size
        # 임시 파일 생성 및 메모리 매핑
        test_scores_path = os.path.join(self.model_save_path, f"{self.dataset}_{self.study_name}_test_scores.npy")
        test_scores_memmap = np.memmap(test_scores_path, dtype='float32', mode='w+',
                                    shape=(total_samples-(self.win_size-1), self.win_size, len(target_columns)))
        print(f'test_score_mem.shape : {test_scores_memmap.shape}')
        zscore_threshold = self.zscore_threshold  # z-score 임계값 설정 (예: 2 또는 3)
        #window_size = self.win_size  # 슬라이딩 윈도우 크기

        with torch.no_grad():
            for i, input_data in enumerate(tqdm(self.test_loader, desc="Test Loader")):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)
                loss = criterion(input, output)  # 각 칼럼별로 loss 계산

                series_loss, prior_loss = 0.0, 0.0
                for u in range(len(prior)):
                    norm_P = torch.clamp((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)), min=1e-10)
                    if u == 0:
                        series_loss += (torch.mean(my_kl_loss(series[u], 
                                            norm_P.detach())) 
                                            +
                                        torch.mean(my_kl_loss(norm_P.detach(),
                                            series[u])))*temperature
                        prior_loss += torch.mean(bhattacharyya_distance(series[u], norm_P))*temperature
                    else:
                        series_loss += (torch.mean(my_kl_loss(series[u], 
                                            norm_P.detach())) 
                                            +
                                        torch.mean(my_kl_loss(norm_P.detach(),
                                            series[u])))*temperature
                        prior_loss += torch.mean(bhattacharyya_distance(series[u].detach(), norm_P))*temperature

                metric = torch.softmax((-series_loss - prior_loss * self.prior_weight), dim=-1).unsqueeze(-1).repeat(1, 1, loss.shape[2])
                score = metric * loss

                # 대상 칼럼에 대한 score만 저장
                target_score = score[:, :, target_columns].cpu().numpy()
                start_idx = i * self.batch_size
                end_idx = min(start_idx + target_score.shape[0], test_scores_memmap.shape[0])
                test_scores_memmap[start_idx:end_idx] = target_score[:end_idx - start_idx]

        print(f"Test scores saved to {test_scores_path}")
        print(f'test_score_shape : {test_scores_memmap.shape}')
        # 메모리 해제 및 정리
        test_scores_memmap.flush()
        test_scores_memmap._mmap.close()
        del test_scores_memmap
        torch.cuda.empty_cache()
        gc.collect()
        
        win_size=self.win_size
        # Test scores memmap 로드
        test_scores_memmap = np.memmap(test_scores_path, dtype='float32', mode='r', shape=(total_samples-(win_size-1), win_size, len(target_columns)))

        prediction_path = os.path.join(self.model_save_path, f"{self.dataset}_{self.study_name}_prediction.npy")
        threshold_path = os.path.join(self.model_save_path, f"{self.dataset}_{self.study_name}_zscore_thresholds.npy")
        rec_pred_path = os.path.join(self.model_save_path, f"{self.dataset}_{self.study_name}_rec_pred.npy")
        rec_pred = np.memmap(rec_pred_path,dtype='float32', mode='w+', shape=(self.total_row,len(target_columns)))




        # 배치 처리 함수 호출
        prediction_memmap=process_batches(test_scores_path=test_scores_path,
                        prediction_path=prediction_path,
                        threshold_path=threshold_path,
                        num_batches=num_batches,
                        batch_size=self.batch_size,
                        win_size=win_size,
                        target_columns=target_columns,
                        zscore_threshold=zscore_threshold,
                        total_samples=total_samples)#,
                        #rec_pred_path=rec_pred_path,
                        #total_row=self.total_row)
        rec_pred=reconstruct_from_windows_memmap(prediction_memmap,self.total_row,win_size,rec_pred_path)
        rec_pred.flush()
        print(f'rec_pred.shape : {rec_pred.shape}')

        agg_res=aggregate_anomalies(prediction=rec_pred,aggregate_unit=self.aggregate_unit, min_length=self.min_anomaly_length)
        agg_res_path=os.path.join(self.model_save_path, f"{self.dataset}_{self.study_name}_agg_res.npy")
        print(f'agg_res.shape : {agg_res.shape}, saved')
        np.save(agg_res_path,agg_res)
#
#
#
#
        ## 평가가 끝난 후 데이터 해제
        #del threshold, test_scores#, predictions
        #torch.cuda.empty_cache()
        #gc.collect()
        


