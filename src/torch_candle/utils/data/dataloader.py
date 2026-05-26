import math
import multiprocessing
import traceback
import random
import numpy as np

class WorkerInfo:
    def __init__(self, id, num_workers, seed, dataset):
        self.id = id
        self.num_workers = num_workers
        self.seed = seed
        self.dataset = dataset

_worker_info = None

def get_worker_info():
    global _worker_info
    return _worker_info

def _worker_loop(dataset, index_queue, result_queue, collate_fn, worker_id, num_workers, seed):
    """Worker process main loop. Fetches data and collates in worker process."""
    global _worker_info
    _worker_info = WorkerInfo(worker_id, num_workers, seed, dataset)
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        while True:
            r = index_queue.get()
            if r is None:  # Shutdown sentinel
                break
            batch_idx, indices = r
            try:
                samples = [dataset[idx] for idx in indices]
                collated = collate_fn(samples)
                result_queue.put((batch_idx, collated, None))
            except Exception as e:
                err_msg = traceback.format_exc()
                result_queue.put((batch_idx, None, (str(e), err_msg)))
    except Exception as e:
        pass

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        # Lazy import to avoid circular dependencies in utils.data package
        if collate_fn is None:
            from . import default_collate
            self.collate_fn = default_collate
        else:
            self.collate_fn = collate_fn

        # Argument validation matching PyTorch
        if self.batch_sampler is not None:
            if (self.batch_size != 1 or self.shuffle or
                    self.sampler is not None or self.drop_last):
                raise ValueError(
                    "batch_sampler option is mutually exclusive with "
                    "batch_size, shuffle, sampler, and drop_last"
                )

        if self.sampler is not None and self.shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")

        if self.num_workers < 0:
            raise ValueError("num_workers option cannot be negative")

        # Initialize Samplers if not explicitly provided
        if self.batch_sampler is None:
            from . import SequentialSampler, RandomSampler, BatchSampler
            if self.sampler is None:
                if self.shuffle:
                    self.sampler = RandomSampler(self.dataset)
                else:
                    self.sampler = SequentialSampler(self.dataset)
            self.batch_sampler = BatchSampler(self.sampler, self.batch_size, self.drop_last)

    def __iter__(self):
        if self.num_workers == 0:
            # Single-process iteration
            for indices in self.batch_sampler:
                samples = [self.dataset[idx] for idx in indices]
                yield self.collate_fn(samples)
        else:
            # Multi-process iteration
            batches = list(self.batch_sampler)
            num_batches = len(batches)
            
            index_queue = multiprocessing.Queue()
            result_queue = multiprocessing.Queue()
            
            # Populate index queue with batches
            for idx, indices in enumerate(batches):
                index_queue.put((idx, indices))
            
            # Append worker shutdown sentinels
            for _ in range(self.num_workers):
                index_queue.put(None)
                
            workers = []
            try:
                # Spawn worker subprocesses
                for i in range(self.num_workers):
                    seed = i + 1000
                    w = multiprocessing.Process(
                        target=_worker_loop,
                        args=(self.dataset, index_queue, result_queue, self.collate_fn, i, self.num_workers, seed)
                    )
                    w.daemon = True
                    w.start()
                    workers.append(w)
                
                received = {}
                next_idx = 0
                while next_idx < num_batches:
                    if next_idx in received:
                        collated, err = received.pop(next_idx)
                        if err:
                            raise RuntimeError(f"DataLoader worker process error:\n{err[1]}")
                        yield collated
                        next_idx += 1
                        continue
                    
                    try:
                        batch_idx, collated, err = result_queue.get()
                        received[batch_idx] = (collated, err)
                    except Exception as e:
                        # Ensure we exit loop on queue error
                        raise RuntimeError(f"Error fetching from result queue: {e}")
            finally:
                # Terminate and clean up all worker processes
                for w in workers:
                    if w.is_alive():
                        w.terminate()
                
                # Drain queue to avoid deadlock on feeder thread
                try:
                    while not result_queue.empty():
                        result_queue.get_nowait()
                except Exception:
                    pass
                    
                for w in workers:
                    w.join()

    def __len__(self):
        return len(self.batch_sampler)

