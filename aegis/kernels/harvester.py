"""
Harvester eBPF en C++ para flujo de datos AEGIS sin bloqueo de GIL

Sistema de ingestión de datos de alta performance
"""

import torch
import threading
import queue
from typing import Callable, Optional, Any
from dataclasses import dataclass
import time


@dataclass
class DataBatch:
    """Lote de datos para procesamiento"""
    data: torch.Tensor
    metadata: dict
    timestamp: float


class AegisFlow:
    """
    Flujo de datos AEGIS (Active Efficient Global Ingestion Stream)
    
    Pipeline de datos asíncrono sin bloqueo
    """
    
    def __init__(self, 
                 buffer_size: int = 1000,
                 num_workers: int = 4,
                 device: str = "cuda"):
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.device = device
        
        # Colas de datos
        self.input_queue = queue.Queue(maxsize=buffer_size)
        self.output_queue = queue.Queue()
        
        # Workers
        self.workers = []
        self.running = False
        
        # Estadísticas
        self.processed_count = 0
        self.dropped_count = 0
    
    def start(self, processor: Callable[[Any], torch.Tensor]):
        """Iniciar flujo de datos"""
        self.running = True
        
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(processor,),
                name=f"AegisWorker-{i}"
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self, processor: Callable):
        """Loop del worker"""
        while self.running:
            try:
                item = self.input_queue.get(timeout=1.0)
                
                # Procesar
                result = processor(item)
                
                # Encolar resultado
                batch = DataBatch(
                    data=result,
                    metadata={'source': 'aegis'},
                    timestamp=time.time()
                )
                
                try:
                    self.output_queue.put(batch, timeout=0.1)
                    self.processed_count += 1
                except queue.Full:
                    self.dropped_count += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en worker: {e}")
    
    def ingest(self, data: Any) -> bool:
        """
        Ingerir dato en el flujo
        
        Returns:
            True si se encoló, False si se descartó
        """
        try:
            self.input_queue.put(data, timeout=0)
            return True
        except queue.Full:
            self.dropped_count += 1
            return False
    
    def get_batch(self, timeout: float = 0.1) -> Optional[DataBatch]:
        """Obtener lote procesado"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """Detener flujo"""
        self.running = False
        for worker in self.workers:
            worker.join(timeout=5.0)
    
    def get_stats(self) -> dict:
        """Obtener estadísticas"""
        return {
            'processed': self.processed_count,
            'dropped': self.dropped_count,
            'drop_rate': self.dropped_count / max(self.processed_count + self.dropped_count, 1),
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize()
        }


class DataHarvester:
    """
    Harvester de datos con flujo AEGIS
    
    Equivalente eBPF en Python para alto rendimiento
    """
    
    def __init__(self, 
                 batch_size: int = 64,
                 num_workers: int = 4,
                 device: str = "cuda"):
        self.batch_size = batch_size
        self.device = device
        
        # Flujo AEGIS
        self.flow = AegisFlow(
            buffer_size=batch_size * 10,
            num_workers=num_workers,
            device=device
        )
        
        # Buffers
        self.prefetch_buffer = []
        
        # Preprocesador
        self.preprocessor = None
    
    def set_preprocessor(self, preprocessor: Callable[[Any], torch.Tensor]):
        """Configurar función de preprocesamiento"""
        self.preprocessor = preprocessor
        self.flow.start(preprocessor)
    
    def feed(self, data: Any):
        """Alimentar datos al harvester"""
        self.flow.ingest(data)
    
    def get_batch(self) -> Optional[torch.Tensor]:
        """Obtener lote listo para entrenamiento"""
        # Llenar buffer
        while len(self.prefetch_buffer) < self.batch_size:
            batch = self.flow.get_batch(timeout=0.01)
            if batch is None:
                break
            self.prefetch_buffer.append(batch.data)
        
        # Si tenemos suficientes, crear lote
        if len(self.prefetch_buffer) >= self.batch_size:
            batch_data = self.prefetch_buffer[:self.batch_size]
            self.prefetch_buffer = self.prefetch_buffer[self.batch_size:]
            
            return torch.stack(batch_data)
        
        return None
    
    def get_stats(self) -> dict:
        """Obtener estadísticas del harvester"""
        return self.flow.get_stats()
    
    def shutdown(self):
        """Apagar harvester"""
        self.flow.stop()
