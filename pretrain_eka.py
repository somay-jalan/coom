from distributed_utils import init_distributed, destroy_distributed
from model_provider import model_provider
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

def main():
    init_distributed()
    model_parallel_cuda_manual_seed(123)

    model = model_provider()
    print(model)
    
    destroy_distributed()

if __name__ == '__main__':
    main()
