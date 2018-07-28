import torch
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple

a = torch.randn(1,4,4).cuda()
b = torch.zeros(a.size()).cuda()

kernel = '''
extern "C"
__global__ void flip(float *dst, const float *src, int w, int total)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i >= total)
      return;
   dst[i] = src[(i / w) * w + (w - (i % w) - 1)];
}
'''


program = Program(kernel, 'flip.cu')
ptx = program.compile()

m = function.Module()
m.load(bytes(ptx.encode()))

