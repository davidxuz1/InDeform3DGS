import os
import re
import io
import torch
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import pyiqa
from collections import defaultdict
from torchvision import transforms
from tqdm import tqdm
import time
import argparse
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import dump, load
import gc
import subprocess
import json
import sys
class FileHandler:
    @staticmethod
    def create_number_mapping(folder_path):
        mapping = defaultdict(list)
        for filename in os.listdir(folder_path):
            match = re.search(r'\d+', filename)
            if match:
                number = str(int(match.group(0)))
                mapping[number].append(filename)
        return mapping

    @staticmethod
    def find_matching_files(exp_folder, gt_folder, mask_folder):
        exp_mapping = FileHandler.create_number_mapping(exp_folder)
        gt_mapping = FileHandler.create_number_mapping(gt_folder)
        mask_mapping = FileHandler.create_number_mapping(mask_folder)
        
        matching_files = []
        for number in exp_mapping.keys():
            if number in gt_mapping and number in mask_mapping:
                matching_files.append({
                    'exp': exp_mapping[number][0],
                    'gt': gt_mapping[number][0],
                    'mask': mask_mapping[number][0]
                })
        return matching_files

class ImageMetricsCalculator:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.niqe_metric = pyiqa.create_metric('niqe').to(self.device)
        self.brisque_metric = pyiqa.create_metric('brisque').to(self.device)
        self.tf_model = None
        self.tf_predict_fn = None

    def predict_mos(self, image_tensor):
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)

        if image_np.ndim == 4:
            image_np = image_np[0]
        if image_np.ndim == 3:
            image_np = np.transpose(image_np, (1, 2, 0))
            
        image_pil = Image.fromarray(image_np)
        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        if self.tf_model is None:
            self._load_tf_model()

        with tf.device('/CPU:0'):
            prediction = self.tf_predict_fn(tf.constant(img_byte_arr))
            mos_score = prediction['output_0'].numpy()
        
        return float(mos_score)

    def _load_tf_model(self):
        with tf.device('/CPU:0'):
            model_handle = 'https://tfhub.dev/google/musiq/paq2piq/1'
            self.tf_model = hub.load(model_handle)
            self.tf_predict_fn = self.tf_model.signatures['serving_default']

    @staticmethod
    def calculate_psnr(img1, img2):
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
        return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()

class MetricsProcessor:
    def __init__(self, exp_folder, gt_folder, mask_folder):
        self.exp_folder = exp_folder
        self.gt_folder = gt_folder
        self.mask_folder = mask_folder
        self.metrics_calculator = ImageMetricsCalculator()
        self.device = self.metrics_calculator.device
        self.transform = transforms.ToTensor()

    def calculate_metrics(self, experiment_id, params, train_time=0, render_time=0, total_time=0):
        try:
            matching_files = FileHandler.find_matching_files(
                self.exp_folder, self.gt_folder, self.mask_folder
            )

            if not matching_files:
                raise ValueError("No se encontraron imágenes coincidentes")

            print(f"Procesando {len(matching_files)} imágenes...")
            all_metrics = []

            for files in tqdm(matching_files):
                try:
                    metrics = self._process_single_image(files)
                    all_metrics.append(metrics)
                except Exception as e:
                    print(f"Error procesando {files['exp']}: {str(e)}")
                    continue

            return self._save_results(all_metrics, experiment_id, train_time, render_time, total_time, params)
        finally:
            # Limpiar memoria después de las métricas
            torch.cuda.empty_cache()
            gc.collect()

    def _process_single_image(self, files):
        exp_path = os.path.join(self.exp_folder, files['exp'])
        gt_path = os.path.join(self.gt_folder, files['gt'])
        mask_path = os.path.join(self.mask_folder, files['mask'])

        exp_img = self.transform(Image.open(exp_path).convert('RGB')).to(self.device)
        gt_img = self.transform(Image.open(gt_path).convert('RGB')).to(self.device)
        mask = self.transform(Image.open(mask_path).convert('L')).to(self.device)

        masked_exp = exp_img * mask
        try:
            return self._calculate_image_metrics(files, exp_img, gt_img, mask, masked_exp)
        finally:
            del exp_img, gt_img, mask, masked_exp
            torch.cuda.empty_cache()
            gc.collect()
    def _calculate_image_metrics(self, files, exp_img, gt_img, mask, masked_exp):
        """Calcula todas las métricas para una imagen individual"""
        try:
            niqe_tensor = self.metrics_calculator.niqe_metric(masked_exp.unsqueeze(0))
            brisque_tensor = self.metrics_calculator.brisque_metric(masked_exp.unsqueeze(0))
            mos = self.metrics_calculator.predict_mos(exp_img)

            niqe = niqe_tensor.cpu().item() if torch.is_tensor(niqe_tensor) else niqe_tensor
            brisque = brisque_tensor.cpu().item() if torch.is_tensor(brisque_tensor) else brisque_tensor

            weighted = 0.5 * mos - 0.25 * niqe - 0.25 * brisque
            psnr_value = self._calculate_psnr_with_cropping(exp_img, gt_img, mask)

            return {
                'image': files['exp'],
                'niqe': niqe,
                'brisque': brisque,
                'mos': mos,
                'weighted': weighted,
                'psnr': psnr_value,
                'final_score': (0.2 * psnr_value + 0.8 * weighted)
            }
            
        except Exception as e:
            print(f"\n[Error Detallado] Fallo en {files['exp']}:")
            print(f"Tipo de niqe: {type(niqe_tensor)}")
            print(f"Tipo de brisque: {type(brisque_tensor)}")
            print(f"Tipo de mos: {type(mos)}")
            raise e

    def _calculate_psnr_with_cropping(self, exp_img, gt_img, mask):
        """Calcula PSNR con cropping si es necesario"""
        try:
            crop_height = 30
            exp_img_psnr = exp_img[:, :-crop_height, :]
            gt_img_psnr = gt_img[:, :-crop_height, :]
            mask_psnr = mask[:, :-crop_height, :]

            inverted_mask = (1 - mask_psnr).to(self.device)
            psnr_tensor = self.metrics_calculator.calculate_psnr(
                (exp_img_psnr * inverted_mask).unsqueeze(0),
                (gt_img_psnr * inverted_mask).unsqueeze(0)
            )

            return psnr_tensor.item() if torch.is_tensor(psnr_tensor) else psnr_tensor
            
        except Exception as e:
            print("\n[Error en PSNR]")
            print(f"Shape exp_img: {exp_img.shape}")
            print(f"Shape gt_img: {gt_img.shape}")
            print(f"Shape mask: {mask.shape}")
            raise e
        
    def _save_results(self, all_metrics, experiment_id, train_time, render_time, total_time, params):
        avg_metrics = self._calculate_averages(all_metrics)
        
        os.makedirs("metric_logs", exist_ok=True)
        metrics_file = os.path.join('metric_logs', f'metrics_results_{experiment_id}.txt')
        
        with open(metrics_file, 'w') as f:
            self._write_metrics_to_file(f, experiment_id, all_metrics, avg_metrics,
                                    train_time, render_time, total_time, params)  # Añadir params
        return all_metrics, avg_metrics

    def _calculate_averages(self, all_metrics):
        return {
            'avg_niqe': np.mean([float(m['niqe']) for m in all_metrics]),
            'avg_brisque': np.mean([float(m['brisque']) for m in all_metrics]),
            'avg_mos': np.mean([float(m['mos']) for m in all_metrics]),
            'avg_weighted': np.mean([float(m['weighted']) for m in all_metrics]),
            'avg_psnr': np.mean([float(m['psnr']) for m in all_metrics]),
            'avg_final_score': np.mean([float(m['final_score']) for m in all_metrics])
        }

    def _write_metrics_to_file(self, file_obj, exp_id, all_metrics, avg_metrics,
                            train_time, render_time, total_time, params=None):
        file_obj.write(f"Experiment {exp_id} Metrics\n")
        file_obj.write(f"Parameters used:\n")
        
        # Escribir parámetros lambda si están disponibles
        if params:
            for key, value in params.items():
                file_obj.write(f"{key}: {value:.2f}\n")
        
        file_obj.write(f"Number of processed images: {len(all_metrics)}\n\n")
   
        file_obj.write("Average metrics:\n")
        for key, value in avg_metrics.items():
            file_obj.write(f"{key}: {value:.6f}\n")
        
        file_obj.write("\nExecution Times:\n")
        file_obj.write(f"Training time: {train_time:.2f} seconds\n")
        file_obj.write(f"Rendering time: {render_time:.2f} seconds\n")
        file_obj.write(f"Total experiment time: {total_time:.2f} seconds\n")
        
        file_obj.write("\nPer-image metrics:\n")
        for metric in all_metrics:
            file_obj.write(f"\nImage: {metric['image']}\n")
            for key, value in metric.items():
                if key != 'image':
                    file_obj.write(f"{key}: {value:.6f}\n")


class ExperimentManager:
    # Mantener los nombres originales para referencia
    LAMBDA_NAMES = [
        'lambda_similarity',
        'lambda_depth_inpaint',
        'lambda_illumination',
        'lambda_diversity',
        'lambda_edge_smoothing',
        'lambda_time_consistency',
        'lambda_noise'
    ]

    # Nuevas constantes para parámetros fijos y ajustables
    FIXED_PARAMS = {
        'lambda_similarity': 0.02,
        'lambda_illumination': 0.0,
        'lambda_time_consistency': 0.0
    }
    
    TUNABLE_LAMBDAS = [
        'lambda_depth_inpaint',
        'lambda_diversity',
        'lambda_edge_smoothing',
        'lambda_noise'
    ]

    def __init__(self, dataset_type):
        self.dataset_type = dataset_type
        # Configurar paths según el dataset
        if self.dataset_type == 'endonerf':
            self.base_exp_folder = "output/endonerf/pulling_fdm_{id}/video/ours_3000/renders"
            self.train_data_path = "data/endonerf/pulling"
            self.base_gt_folder = "base/endonerf/images"
            self.base_mask_folder = "base/endonerf/gt_masks"
        elif self.dataset_type == 'own_video':
            self.base_exp_folder = "output/own_video/pulling_fdm_{id}/video/ours_3000/renders"
            self.train_data_path = "data/own_video"
            self.base_gt_folder = "base/own_video/images"
            self.base_mask_folder = "base/own_video/gt_masks"

        self.checkpoint_file = os.path.join('metric_logs', 'bayesian_checkpoint.pkl')
        self.counter_file = os.path.join('metric_logs', 'bayesian_counter.txt')
        self._init_counter()

    def _init_counter(self):
        """Inicializa o carga el contador de experimentos"""
        if os.path.exists(self.counter_file):
            with open(self.counter_file, 'r') as f:
                self.bayes_counter = int(f.read())
        else:
            self.bayes_counter = 1

    def run_experiments(self, experiment_type):
        if experiment_type == 1:
            experiments = self.create_custom_experiment()
            self._run_standard_experiments(experiments)
        elif experiment_type == 2:
            experiments = self.create_individual_lambda_experiments()
            self._run_standard_experiments(experiments)
        elif experiment_type == 3:
            self._run_bayesian_optimization()
        else:
            raise ValueError("Tipo de experimento no válido")

    def _run_standard_experiments(self, experiments):
        for i, params in enumerate(experiments, start=1):
            self._run_single_experiment(params, i)

    def _run_bayesian_optimization(self, n_calls=150, n_random_starts=10, continue_opt=False, xi=0.1):
        """Optimización Bayesiana modificada con parámetros fijos"""
        os.makedirs('metric_logs', exist_ok=True)
        
        # Cargar estado previo si existe
        if continue_opt and os.path.exists(self.checkpoint_file):
            result = load(self.checkpoint_file)
            print(f"Optimización cargada - Iteraciones previas: {len(result.x_iters)}")
        else:
            result = None

        # Definir solo los parámetros a optimizar
        dimensions = [Real(0.0, 1.0, name=name, prior='uniform') 
                     for name in self.TUNABLE_LAMBDAS]

        # Configurar y ejecutar la optimización
        result = gp_minimize(
            func=self.objective_function,
            dimensions=dimensions,
            n_calls=n_calls,
            n_initial_points=n_random_starts,
            x0=result.x_iters if result else None,
            y0=result.func_vals if result else None,
            callback=self._save_checkpoint,
            random_state=42,
            verbose=True,
            acq_func="EI",
            xi=xi
        )

        # Guardar resultados finales
        self._save_final_results(result)
        return result


    def objective_function(self, params):
        """Función objetivo con parámetros fijos integrados"""
        # Combinar parámetros fijos y ajustables
        full_params = self.FIXED_PARAMS.copy()
        full_params.update({
            name: round(value, 2)
            for name, value in zip(self.TUNABLE_LAMBDAS, params)
        })
        
        experiment_id = f"bayes_{self.bayes_counter}"
        self.bayes_counter += 1
        self._update_counter_file()

        params_file = os.path.join('metric_logs', f'params_{experiment_id}.json')
        try:
            with open(params_file, 'w') as f:
                json.dump(full_params, f)

            # cmd = ["python", sys.argv[0], "--type", "subprocess", "--exp_id", experiment_id, "--params", params_file]
            cmd = ["python", sys.argv[0], 
            "--dataset", self.dataset_type,  # Add this line
            "--type", "subprocess", 
            "--exp_id", experiment_id, 
            "--params", params_file]
            subprocess.run(cmd, check=True, timeout=7200)

            return self._read_experiment_score(experiment_id)
        finally:
            if os.path.exists(params_file):
                os.remove(params_file)

    def _read_experiment_score(self, experiment_id):
        """Lee el score del archivo de métricas"""
        metrics_file = os.path.join('metric_logs', f'metrics_results_{experiment_id}.txt')
        try:
            with open(metrics_file, 'r') as f:
                for line in f:
                    if "avg_final_score" in line:
                        return -float(line.split(":")[1].strip())
            return 1000  # Valor por defecto si no encuentra el score
        except FileNotFoundError:
            return 1000

    def _save_checkpoint(self, result):
        """Guarda el estado actual de la optimización y muestra información detallada"""
        try:
            # Guardar el checkpoint
            dump(result, self.checkpoint_file)
            
            # Mensaje de confirmación con información clave
            print(f"\n**Checkpoint guardado correctamente**")
            print(f"   - Ruta: {os.path.abspath(self.checkpoint_file)}")
            print(f"   - Tamaño: {os.path.getsize(self.checkpoint_file)/1024:.2f} KB")
            print(f"   - Iteraciones completadas: {len(result.x_iters)}")
            print(f"   - Mejor score actual: {-result.fun if result.fun < 1000 else 'N/A':.2f}")
            
        except Exception as e:
            print(f"\n**Error crítico al guardar checkpoint**")
            print(f"   - Tipo de error: {type(e).__name__}")
            print(f"   - Detalles: {str(e)}")
            print(f"   - Ruta intentada: {self.checkpoint_file}")

    def _print_current_progress(self, result):
        """Muestra el progreso actual en la terminal"""
        current_iter = len(result.func_vals)
        best_score = -result.fun if result.fun < 1000 else "N/A"
        
        print(f"\n{'='*40}")
        print(f"Iteración completada: {current_iter}")
        print(f"Mejor score actual: {best_score}")
        if best_score != "N/A":
            print("Mejores parámetros actuales:")
            for name, value in zip(self.LAMBDA_NAMES, result.x):
                print(f"  - {name}: {value:.2f}")
        print(f"{'='*40}\n")

    def _update_counter_file(self):
        """Actualiza el archivo de contador de forma atómica"""
        with open(self.counter_file, 'w') as f:
            f.write(str(self.bayes_counter))

    def _save_final_results(self, result):
        """Guarda los resultados incluyendo parámetros fijos"""
        # Resultados de la optimización
        dump(result, self.checkpoint_file)
        
        # Combinar parámetros fijos con los optimizados
        best_params = self.FIXED_PARAMS.copy()
        best_params.update({
            name: round(value, 2)
            for name, value in zip(self.TUNABLE_LAMBDAS, result.x)
        })
        
        with open(os.path.join('metric_logs', 'best_params.txt'), 'w') as f:
            f.write("Parámetros óptimos encontrados:\n")
            for name in self.LAMBDA_NAMES:
                f.write(f"{name}: {best_params.get(name, 'Fijado')}\n")
        
        print("\nOptimización completada! Resultados guardados en:")
        print(f"  - Checkpoint principal: {self.checkpoint_file}")
        print(f"  - Parámetros óptimos: metric_logs/best_params.txt")

    def _parse_score_from_file(self, metrics_file):
        """Extrae el score final del archivo de métricas"""
        try:
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if "avg_final_score" in line:
                        return -float(line.split(":")[1].strip())
                return 1000  # Si no encuentra el score
        except FileNotFoundError:
            return 1000
    def run_subprocess_experiment(self, exp_id, params_file):
        """Método para ejecutar desde línea de comandos en subproceso"""
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        try:
            # Ejecutar experimento
            _, avg_metrics = self._run_single_experiment(params, exp_id)
            return -avg_metrics['avg_final_score']
        except Exception as e:
            print(f"Error en subproceso {exp_id}: {str(e)}")
            return 1000

    def _print_progress(self, result):
        current_iter = len(result.func_vals)
        best_score = -result.fun if result.fun != 1000 else "N/A"
        
        print(f"\n{'='*40}")
        print(f"Iteración: {current_iter}")
        print(f"Mejor score actual: {best_score}")
        
        if best_score != "N/A":
            print("Mejores parámetros hasta ahora:")
            for i, name in enumerate(self.LAMBDA_NAMES):
                print(f"{name}: {result.x[i]:.2f}")
        print(f"{'='*40}\n")

    def save_optimization_state(self, result, checkpoint_file):
        """Guarda solo el estado de optimización sin el contador"""
        dump(result, checkpoint_file)
        # Guardar contador en archivo separado
        with open(self.checkpoint_counter_file, 'w') as f:
            f.write(str(self.bayes_counter))


    def load_optimization_state(self, checkpoint_file):
        """Carga el estado y el contador por separado"""
        if os.path.exists(checkpoint_file):
            result = load(checkpoint_file)
            # Cargar contador si existe
            if os.path.exists(self.checkpoint_counter_file):
                with open(self.checkpoint_counter_file, 'r') as f:
                    self.bayes_counter = int(f.read()) + 1  # +1 para evitar duplicados
            return result
        return None

    def _run_single_experiment(self, params, experiment_id):
        try:
            # Limpieza inicial de memoria
            torch.cuda.empty_cache()
            gc.collect()
            
            # Configurar TensorFlow para usar solo CPU
            tf.config.set_visible_devices([], 'GPU')
            
            # Crear configuración
            config_file = self.create_config_file(params, experiment_id)
            if self.dataset_type == 'endonerf':
                expname = f"endonerf/pulling_fdm_{experiment_id}"
            if self.dataset_type == 'own_video':
                expname = f"own_video/pulling_fdm_{experiment_id}"
            
            # Limitar memoria de GPU para PyTorch
            torch.cuda.set_per_process_memory_fraction(0.7)
            
            # Ejecutar entrenamiento y renderizado en subprocesos
            train_time = self._run_training(config_file, expname, experiment_id)
            render_time = self._run_rendering(expname, config_file, experiment_id)
            
            # Evaluar resultados
            return self._evaluate_experiment(
                experiment_id, params, train_time, render_time, 
                train_time + render_time
            )
            
        except Exception as e:
            print(f"Error crítico en experimento {experiment_id}: {str(e)}")
            return None
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def create_config_file(self, params, experiment_id):
        os.makedirs("arguments/endonerf", exist_ok=True)
        if self.dataset_type == 'endonerf':

            config_content = f"""ModelParams = dict(
    extra_mark = 'endonerf',
    camera_extent = 10,
    use_inpainting = True
)
DatasetParams = dict(
    # img_width = 224,
    # img_height = 224, 
    img_width = 640, # si endonerf
    img_height = 512,
    downsample = 1.0,
    test_every = 8,
    #dataset_dir = 'data/own_video'
    dataset_dir = 'data/endonerf/pulling'
)
OptimizationParams = dict(
    coarse_iterations = 0,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    iterations = 3000,
    percent_dense = 0.01,
    opacity_reset_interval = 3000,
    position_lr_max_steps = 4000,
    prune_interval = 3000,
    lambda_similarity = {round(params['lambda_similarity'], 2)},
    lambda_depth_inpaint = {round(params['lambda_depth_inpaint'], 2)},
    lambda_illumination = {round(params['lambda_illumination'], 2)},
    lambda_diversity = {round(params['lambda_diversity'], 2)},
    lambda_edge_smoothing = {round(params['lambda_edge_smoothing'], 2)},
    lambda_time_consistency = {round(params['lambda_time_consistency'], 2)},
    lambda_noise = {round(params['lambda_noise'], 2)}
)

ModelHiddenParams = dict(
    curve_num = 17,
    ch_num = 10,
    init_param = 0.01,
)"""
        if self.dataset_type == 'own_video':

            config_content = f"""ModelParams = dict(
    extra_mark = 'endonerf',
    camera_extent = 10,
    use_inpainting = True
)
DatasetParams = dict(
    img_width = 224,
    img_height = 224, 
    downsample = 1.0,
    test_every = 8,
    dataset_dir = 'data/own_video'
)
OptimizationParams = dict(
    coarse_iterations = 0,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    iterations = 3000,
    percent_dense = 0.01,
    opacity_reset_interval = 3000,
    position_lr_max_steps = 4000,
    prune_interval = 3000,
    lambda_similarity = {round(params['lambda_similarity'], 2)},
    lambda_depth_inpaint = {round(params['lambda_depth_inpaint'], 2)},
    lambda_illumination = {round(params['lambda_illumination'], 2)},
    lambda_diversity = {round(params['lambda_diversity'], 2)},
    lambda_edge_smoothing = {round(params['lambda_edge_smoothing'], 2)},
    lambda_time_consistency = {round(params['lambda_time_consistency'], 2)},
    lambda_noise = {round(params['lambda_noise'], 2)}
)

ModelHiddenParams = dict(
    curve_num = 17,
    ch_num = 10,
    init_param = 0.01,
)"""
        config_file = f"arguments/endonerf/default_{experiment_id}.py"
        with open(config_file, 'w') as f:
            f.write(config_content)
        return config_file

    # def create_individual_lambda_experiments(self):
    #     experiments = []
    #     for lambda_name in self.LAMBDA_NAMES:
    #         for value in [round(x * 0.1, 1) for x in range(1, 11)]:
    #             params = {name: 0.0 for name in self.LAMBDA_NAMES}
    #             params[lambda_name] = value
    #             experiments.append(params)
    #     return experiments

    def create_individual_lambda_experiments(self):
        experiments = []
        lambda_name = 'lambda_similarity'
        values = [0.07, 0.08, 0.09, 0.1]
        
        for value in values:
            params = {name: 0.0 for name in self.LAMBDA_NAMES}
            params[lambda_name] = value
            experiments.append(params)
        
        return experiments


    
    # def create_individual_lambda_experiments(self):
    #     experiments = []
    #     # Probar solo lambda_similarity
    #     lambda_name = 'lambda_similarity'
    #     values = [round(x * 0.01, 2) for x in range(0, 11)]  # 0.00 a 0.10
        
    #     for value in values:
    #         params = {name: 0.0 for name in self.LAMBDA_NAMES}
    #         params[lambda_name] = value
    #         experiments.append(params)
        
    #     return experiments

    # def create_custom_experiment(self):
    #     # Lista de configuraciones personalizadas que quieres probar
    #     custom_experiments = [
    #         {
    #         'lambda_similarity': 0.02,
    #         'lambda_depth_inpaint': 0.91,
    #         'lambda_illumination': 0.00,
    #         'lambda_diversity': 0.22,
    #         'lambda_edge_smoothing': 0.26,
    #         'lambda_time_consistency': 0.0,
    #         'lambda_noise': 1.0
    #         }
    #     ]
    #     return custom_experiments
    
    def create_custom_experiment(self, lambda_values):
        # Crear un diccionario con los valores de lambda proporcionados
        custom_experiment = {
            'lambda_similarity': lambda_values[0],
            'lambda_depth_inpaint': lambda_values[1],
            'lambda_illumination': lambda_values[2],
            'lambda_diversity': lambda_values[3],
            'lambda_edge_smoothing': lambda_values[4],
            'lambda_time_consistency': lambda_values[5],
            'lambda_noise': lambda_values[6]
        }
        return [custom_experiment]

    def _run_training(self, config_file, expname, exp_id):
        print(f"\nIniciando entrenamiento para experimento {exp_id}")
        train_start = time.time()
        
        # Usar el path correspondiente al dataset
        train_cmd = [
            "python", "train.py",
            "-s", self.train_data_path,
            "--expname", expname,
            "--configs", config_file
        ]

        
        try:
            # Ejecutar mostrando output en tiempo real
            process = subprocess.Popen(
                train_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Mostrar líneas conforme se generen
            for line in process.stdout:
                print(line, end='')
                
            process.wait()
            
        except Exception as e:
            print(f"Error en entrenamiento ({exp_id}): {str(e)}")
            return 0
        
        return time.time() - train_start


    def _run_rendering(self, expname, config_file, exp_id):
        print(f"\nIniciando renderizado para experimento {exp_id}")
        render_start = time.time()
        render_cmd = [
            "python", "render.py",
            "--model_path", f"output/{expname}",
            "--skip_train",
            "--reconstruct_test",
            "--configs", config_file
        ]
        
        try:
            # Ejecutar mostrando output en tiempo real
            process = subprocess.Popen(
                render_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Mostrar líneas conforme se generen
            for line in process.stdout:
                print(line, end='')
                
            process.wait()
            
        except Exception as e:
            print(f"Error en renderizado ({exp_id}): {str(e)}")
            return 0
        
        return time.time() - render_start
    
    def _evaluate_experiment(self, exp_id, params, train_time, render_time, total_time):
        exp_folder = self.base_exp_folder.format(id=exp_id)
        metrics_processor = MetricsProcessor(exp_folder, self.base_gt_folder, self.base_mask_folder)
        
        try:
            metrics, averages = metrics_processor.calculate_metrics(
                exp_id, params, train_time, render_time, total_time 
            )
            self._save_experiment_params(exp_id, params)
            print(f"Evaluación completada para experimento {exp_id}")
            return exp_id, averages
        except Exception as e:
            print(f"Error evaluando experimento {exp_id}: {str(e)}")
            return exp_id, None

    def _save_experiment_params(self, exp_id, params):
        with open(os.path.join('metric_logs', f'metrics_results_{exp_id}.txt'), 'a') as f:
            f.write("\nExperiment Parameters:\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ejecuta experimentos de NeRF')
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['endonerf', 'own_video'],
                      help='Tipo de dataset a utilizar')
    parser.add_argument('--type', type=str, required=True,
                      choices=['1', '2', '3', 'subprocess'],
                      help='Tipo de experimento')
    parser.add_argument('--continue', dest='continue_opt', action='store_true',
                      help='Continúa una optimización Bayesiana previa')
    parser.add_argument('--n_calls', type=int, default=300,
                      help='Número total de iteraciones a realizar')
    parser.add_argument('--exp_id', type=str,
                      help='ID del experimento para subprocesos')
    parser.add_argument('--params', type=str,
                      help='Archivo JSON con parámetros para subprocesos')
    # Nuevos argumentos para los valores lambda
    parser.add_argument('--lambdas', type=float, nargs=7, 
                      help='Valores para los 7 lambdas en orden: similarity, depth_inpaint, illumination, diversity, edge_smoothing, time_consistency, noise')
    args = parser.parse_args()
    
    # Asegurar que el directorio de métricas existe
    os.makedirs('metric_logs', exist_ok=True)
    
    experiment_manager = ExperimentManager(args.dataset)
    
    if args.type == 'subprocess':
        if not args.exp_id or not args.params:
            raise ValueError("Se requieren --exp_id y --params para modo subproceso")
        
        # Verificar existencia del archivo de parámetros
        if not os.path.exists(args.params):
            raise FileNotFoundError(f"Archivo de parámetros no encontrado: {args.params}")
        
        score = experiment_manager.run_subprocess_experiment(args.exp_id, args.params)
        print(f"Resultado del subproceso {args.exp_id}: {score}")
    elif args.type == '3':
        experiment_manager._run_bayesian_optimization(
            continue_opt=args.continue_opt,
            n_calls=args.n_calls,
            xi=0.1
        )
    elif args.type == '1':
        if args.lambdas:
            experiments = experiment_manager.create_custom_experiment(args.lambdas)
        else:
            # Valores por defecto si no se proporcionan lambdas
            default_lambdas = [0.02, 0.91, 0.00, 0.22, 0.26, 0.0, 1.0]
            experiments = experiment_manager.create_custom_experiment(default_lambdas)
        experiment_manager._run_standard_experiments(experiments)
    else:
        experiment_manager.run_experiments(int(args.type))



## Continuar con 300 iteraciones adicionales y xi=0.1
# python iterar_complete.py --type 3 --continue --n_calls 300 --xi 0.1


# import os
# import random
# import re
# import io
# import torch
# import numpy as np
# import tensorflow as tf
# import tensorflow_hub as hub
# from PIL import Image
# import pyiqa
# from collections import defaultdict
# from torchvision import transforms
# from tqdm import tqdm
# import time

# # Inicializar métricas
# niqe_metric = pyiqa.create_metric('niqe')
# brisque_metric = pyiqa.create_metric('brisque')

# def predict_mos(image_tensor):
#     """Predicts Mean Opinion Score (MOS) for an image using MusIQ model."""
#     image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)

#     if image_np.ndim == 4:
#         image_np = image_np[0]
#     if image_np.ndim == 3:
#         image_np = np.transpose(image_np, (1, 2, 0))
#     image_pil = Image.fromarray(image_np)
    
#     img_byte_arr = io.BytesIO()
#     image_pil.save(img_byte_arr, format='PNG')
#     img_byte_arr = img_byte_arr.getvalue()
    
#     if not hasattr(predict_mos, 'model'):
#         with tf.device('/CPU:0'):
#             model_handle = 'https://tfhub.dev/google/musiq/paq2piq/1'
#             predict_mos.model = hub.load(model_handle)
#             predict_mos.predict_fn = predict_mos.model.signatures['serving_default']
    
#     with tf.device('/CPU:0'):
#         prediction = predict_mos.predict_fn(tf.constant(img_byte_arr))
#         mos_score = prediction['output_0'].numpy()
    
#     return float(mos_score)

# def calculate_psnr(img1, img2):
#     """Calculate PSNR between two images."""
#     mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
#     return 20 * torch.log10(1.0 / torch.sqrt(mse))

# def create_number_mapping(folder_path):
#     """Create a mapping of numbers to filenames."""
#     mapping = defaultdict(list)
#     for filename in os.listdir(folder_path):
#         match = re.search(r'\d+', filename)
#         if match:
#             number = str(int(match.group(0)))
#             mapping[number].append(filename)
#     return mapping

# def find_matching_files(exp_folder, gt_folder, mask_folder):
#     """Find corresponding files across the three folders."""
#     exp_mapping = create_number_mapping(exp_folder)
#     gt_mapping = create_number_mapping(gt_folder)
#     mask_mapping = create_number_mapping(mask_folder)
    
#     matching_files = []
#     for number in exp_mapping.keys():
#         if number in gt_mapping and number in mask_mapping:
#             matching_files.append({
#                 'exp': exp_mapping[number][0],
#                 'gt': gt_mapping[number][0],
#                 'mask': mask_mapping[number][0]
#             })
#     return matching_files

# def calculate_metrics(exp_folder, gt_folder, mask_folder, experiment_id, train_time=0, render_time=0, total_time=0):
#     """Calculate metrics for matching images across folders."""
#     matching_files = find_matching_files(exp_folder, gt_folder, mask_folder)
    
#     if not matching_files:
#         raise ValueError("No se encontraron imágenes coincidentes en las tres carpetas")
    
#     print(f"Procesando {len(matching_files)} imágenes coincidentes...")
    
#     transform = transforms.ToTensor()
#     all_metrics = []
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#     for files in tqdm(matching_files):
#         try:
#             exp_path = os.path.join(exp_folder, files['exp'])
#             gt_path = os.path.join(gt_folder, files['gt'])
#             mask_path = os.path.join(mask_folder, files['mask'])
            
#             exp_img = transform(Image.open(exp_path).convert('RGB')).to(device)
#             gt_img = transform(Image.open(gt_path).convert('RGB')).to(device)
#             mask = transform(Image.open(mask_path).convert('L')).to(device)
            
#             masked_exp = exp_img * mask
            
#             niqe = niqe_metric(masked_exp.unsqueeze(0).to(device))
#             brisque = brisque_metric(masked_exp.unsqueeze(0).to(device))
#             mos = predict_mos(exp_img.to(device))
#             weighted = 0.5 * mos - 0.25 * niqe.cpu() - 0.25 * brisque.cpu()
            
#             i=1
#             # Para PSNR, usar versiones croppeadas si i=1
#             if i == 1:
#                 crop_height = 30  # Ajustar según tamaño de marca de agua
#                 exp_img_psnr = exp_img[:, :-crop_height, :]
#                 gt_img_psnr = gt_img[:, :-crop_height, :]
#                 mask_psnr = mask[:, :-crop_height, :]
                
#                 # # Guardar imagen de debug para verificar cropping
#                 # if files['exp'] == matching_files[0]['exp']:
#                 #     debug_img = torch.cat([exp_img_psnr, gt_img_psnr], dim=2)
#                 #     debug_img = (debug_img.cpu().numpy() * 255).astype(np.uint8)
#                 #     debug_img = np.transpose(debug_img, (1, 2, 0))
#                 #     Image.fromarray(debug_img).save("debug_images/cropped_comparison_psnr.png")
#             else:
#                 exp_img_psnr = exp_img
#                 gt_img_psnr = gt_img
#                 mask_psnr = mask

#             # Calcular PSNR con imágenes croppeadas si corresponde
#             inverted_mask = (1 - mask_psnr).to(device)
#             psnr_value = calculate_psnr(
#                 (exp_img_psnr * inverted_mask).unsqueeze(0),
#                 (gt_img_psnr * inverted_mask).unsqueeze(0)
#             )
            
#             metrics = {
#                 'image': files['exp'],
#                 'niqe': niqe.cpu().item(),
#                 'brisque': brisque.cpu().item(),
#                 'mos': mos,
#                 'weighted': weighted,
#                 'psnr': psnr_value.cpu().item(),
#                 'final_score': (0.2 * psnr_value.cpu() + 0.8 * weighted).item()
#             }
#             all_metrics.append(metrics)
            
#         except Exception as e:
#             print(f"Error procesando {files['exp']}: {str(e)}")
#             continue

#     if not all_metrics:
#         raise ValueError("No se pudieron procesar métricas para ninguna imagen")
    
#     avg_metrics = {
#         'avg_niqe': float(np.mean([float(m['niqe']) for m in all_metrics])),
#         'avg_brisque': float(np.mean([float(m['brisque']) for m in all_metrics])),
#         'avg_mos': float(np.mean([float(m['mos']) for m in all_metrics])),
#         'avg_weighted': float(np.mean([float(m['weighted']) for m in all_metrics])),
#         'avg_psnr': float(np.mean([float(m['psnr']) for m in all_metrics])),
#         'avg_final_score': float(np.mean([float(m['final_score']) for m in all_metrics]))
#     }
    
#     # Guardar métricas en archivo específico del experimento
#     os.makedirs("metric_logs", exist_ok=True)
#     metrics_file = os.path.join('metric_logs', f'metrics_results_{experiment_id}.txt')
#     with open(metrics_file, 'w') as f:
#         f.write(f"Experiment {experiment_id} Metrics\n")
#         f.write(f"Parameters used:\n")
#         f.write(f"Number of processed images: {len(all_metrics)}\n\n")
        
#         f.write("Average metrics:\n")
#         for key, value in avg_metrics.items():
#             f.write(f"{key}: {float(value):.4f}\n")
        
#         # Escribir tiempos aquí, justo después de avg_metrics
#         f.write("\nExecution Times:\n")
#         f.write(f"Training time: {train_time:.2f} seconds\n")
#         f.write(f"Rendering time: {render_time:.2f} seconds\n")
#         f.write(f"Total experiment time: {total_time:.2f} seconds\n")

        
#         f.write("\nPer-image metrics:\n")
#         for metric in all_metrics:
#             f.write(f"\nImage: {metric['image']}\n")
#             for key, value in metric.items():
#                 if key != 'image':
#                     f.write(f"{key}: {float(value):.4f}\n")
#     return all_metrics, avg_metrics

# def create_default_py(params, experiment_id):
#     """Crea un archivo default.py con los parámetros especificados."""
#     os.makedirs("arguments/endonerf", exist_ok=True)
    
#     config_content = f"""ModelParams = dict(
#     extra_mark = 'endonerf',
#     camera_extent = 10
# )

# OptimizationParams = dict(
#     coarse_iterations = 0,
#     deformation_lr_init = 0.00016,
#     deformation_lr_final = 0.0000016,
#     deformation_lr_delay_mult = 0.01,
#     iterations = 3000,
#     percent_dense = 0.01,
#     opacity_reset_interval = 3000,
#     position_lr_max_steps = 4000,
#     prune_interval = 3000,
#     lambda_similarity = {params['lambda_similarity']},
#     lambda_depth_inpaint = {params['lambda_depth_inpaint']},
#     lambda_illumination = {params['lambda_illumination']},
#     lambda_diversity = {params['lambda_diversity']},
#     lambda_edge_smoothing = {params['lambda_edge_smoothing']},
#     lambda_time_consistency = {params['lambda_time_consistency']}   
# )

# ModelHiddenParams = dict(
#     curve_num = 17,
#     ch_num = 10,
#     init_param = 0.01,
# )"""
#     config_file = f"arguments/endonerf/default_{experiment_id}.py"
#     with open(config_file, 'w') as f:
#         f.write(config_content)
#     return config_file

# def create_individual_lambda_experiments():
#     """Crea experimentos variando cada lambda individualmente mientras mantiene los demás en 0."""
#     experiments = []
#     lambda_names = [
#         'lambda_similarity',
#         'lambda_depth_inpaint',
#         'lambda_illumination',
#         'lambda_diversity',
#         'lambda_edge_smoothing',
#         'lambda_time_consistency'
#     ]
    
#     # Para cada lambda
#     for lambda_name in lambda_names:
#         # Variar de 0.1 a 1.0
#         for value in [round(x * 0.1, 1) for x in range(1, 11)]:
#             # Crear diccionario con todos los lambdas en 0.0
#             params = {name: 0.0 for name in lambda_names}
#             # Establecer el valor actual para el lambda que estamos probando
#             params[lambda_name] = value
#             experiments.append(params)

#     # # Variar solo lambda_diversity de 0.1 a 1.0
#     # for value in [round(x * 0.1, 1) for x in range(1, 11)]:
#     #     # Crear diccionario con todos los lambdas en 0.0
#     #     params = {name: 0.0 for name in lambda_names}
#     #     # Establecer el valor actual para lambda_diversity
#     #     params['lambda_diversity'] = value
#     #     experiments.append(params)

#     return experiments

# # Para experimentos individuales
# def create_custom_experiment():
#     """Crea un experimento con valores específicos para cada lambda."""
#     params = {
#         'lambda_similarity': 0.0,
#         'lambda_depth_inpaint': 0.0,
#         'lambda_illumination': 0.0,
#         'lambda_diversity': 0.0,
#         'lambda_edge_smoothing': 0.0,
#         'lambda_time_consistency': 0.0
#     }
#     return [params]  # Return as list to maintain compatibility with run_experiments

# def run_experiments():
#     """Ejecuta los experimentos con diferentes configuraciones."""
#     # Obtener los experimentos de lambda individuales [0.1, 1] con variación 0.1
#     # experiments = create_individual_lambda_experiments()

#     # # Para experimentos individuales
#     experiments = create_custom_experiment()

#     for i, params in enumerate(experiments, start=1):
#         # Crear archivo de configuración
#         config_file = create_default_py(params, i)
#         expname = f"endonerf/pulling_fdm_{i}"
        
#         start_time = time.time()

#         # Entrenar
#         train_start = time.time()
#         train_cmd = f"python train.py -s data/endonerf_full_datasets/pulling_soft_tissues --expname {expname} --configs {config_file}"
#         print(f"\nRunning experiment {i}:")
#         print(f"Parameters: {params}")
#         print(f"Config file: {config_file}")
#         print(f"Output directory: {expname}")
#         print(f"Train command: {train_cmd}\n")
#         os.system(train_cmd)
#         train_time = time.time() - train_start

#         # Renderizar
#         render_start = time.time()
#         render_cmd = f"python render.py --model_path output/endonerf/pulling_fdm_{i} --skip_train --reconstruct_test --configs {config_file}"
#         print(f"\nRunning render for experiment {i}:")
#         print(f"Render command: {render_cmd}\n")
#         os.system(render_cmd)
#         render_time = time.time() - render_start
        
#         total_time = time.time() - start_time
        
#         # Evaluar
#         exp_folder = f'output/endonerf/pulling_fdm_{i}/video/ours_3000/renders'
#         gt_folder = 'base/images'
#         mask_folder = 'base/gt_masks'
        
#         print(f"\nEvaluating experiment {i}:")
#         try:
#             metrics, averages = calculate_metrics(exp_folder, gt_folder, mask_folder, i, 
#                                     train_time=train_time, 
#                                     render_time=render_time, 
#                                     total_time=total_time)
#             print(f"Evaluation complete for experiment {i}")
            
#             # Guardar parámetros junto con métricas
#             with open(os.path.join('metric_logs', f'metrics_results_{i}.txt'), 'a') as f:
#                 f.write("\nExperiment Parameters:\n")
#                 for key, value in params.items():
#                     f.write(f"{key}: {value}\n")
                
#         except Exception as e:
#             print(f"Error evaluating experiment {i}: {str(e)}")
        
#         print(f"Finished experiment {i} (train + render + evaluate)")
#         print("-" * 50)

# if __name__ == "__main__":
#     run_experiments()
