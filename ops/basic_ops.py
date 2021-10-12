# import torch


# class SegmentConsensus(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, input_tensor):
#         result = input_tensor.mean(dim=1, keepdim=True)
#         ctx.save_for_backward(input_tensor)

#         return result
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         input_tensor, = ctx.saved_tensors # a tuple: (1,)
#         grad_in = grad_output.expand(input_tensor.size()) / float(input_tensor.size(1))

#         return grad_in


# class ConsensusModule(torch.nn.Module):

#     def __init__(self, consensus_type, dim=1):
#         super(ConsensusModule, self).__init__()

#     def forward(self, input):
#         return SegmentConsensus().apply(input)


# import torch


# class Identity(torch.nn.Module):
#     def forward(self, input):
#         return input


# class SegmentConsensus(torch.autograd.Function):

#     def __init__(self, consensus_type, dim=1):
#         self.consensus_type = consensus_type
#         self.dim = dim
#         self.shape = None

#     def forward(self, input_tensor):
#         self.shape = input_tensor.size()
#         if self.consensus_type == 'avg':
#             output = input_tensor.mean(dim=self.dim, keepdim=True)
#         elif self.consensus_type == 'identity':
#             output = input_tensor
#         else:
#             output = None

#         return output

#     def backward(self, grad_output):
#         if self.consensus_type == 'avg':
#             grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
#         elif self.consensus_type == 'identity':
#             grad_in = grad_output
#         else:
#             grad_in = None

#         return grad_in


# class ConsensusModule(torch.nn.Module):

#     def __init__(self, consensus_type, dim=1):
#         super(ConsensusModule, self).__init__()
#         self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
#         self.dim = dim

#     def forward(self, input):
#         return SegmentConsensus(self.consensus_type, self.dim)(input)


import torch


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)
