import numpy as np

class tensor():
    def __init__(self, data, _children=(), requires_grad=False):
        self.data = np.array(data)
        self._prev = set(_children)
        self.grad = 0.0
        self._backward = lambda: None
        self.requires_grad = requires_grad

    def __repr__(self):
        return f'tensor({self.data}, requires_grad={self.requires_grad})'
    
    @property
    def shape(self):
        return self.data.shape
    
    def __add__(self, other):
        other = other if isinstance(other, tensor) else tensor(other)
        # assert self.shape == other.shape, "broadcasting is not supported"
        other.data = other.data if self.shape == other.shape else np.broadcast_to(other.data, self.shape)

        out = tensor(self.data + other.data, (self, other))
        
        # Calculate gradients for __add__
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, tensor) else tensor(other)
        # assert self.shape == other.shape, "broadcasting is not supported"
        other.data = other.data if self.shape == other.shape else np.broadcast_to(other.data, self.shape)

        out = tensor(self.data * other.data, (self, other))

        # Calculate gradients for __mul__
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "`other` must be an int or float"
        out = tensor(self.data**other, (self,))

        # Calculate gradients for __pow__
        def _backward():
            self.grad += other * self.data**(other-1) * out.grad
        out._backward = _backward

        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, tensor) else tensor(other)
        # assert self.shape[-1] == self.shape[-2]
        out_data = np.dot(self.data, other.data)
        out = tensor(out_data, (self, other))
        
        # Calculate gradients for matrix multiplication
        def _backward():
            self.grad += np.dot(out.grad, other.data.T)
            other.grad += np.dot(self.data.T, out.grad)
        out._backward = _backward

        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def log(self):
        out_data = np.log(self.data)
        out = tensor(out_data, (self,))

        # Calculate gradients for log()
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        out = tensor(np.tanh(self.data), (self,))

        # Calculate gradients for tanh()
        def _backward():
            self.grad += (1 - self.data**2)*out.grad
        out._backward = _backward

        return out
    
    def sigmoid(self):
        y = 1 / (1 + np.exp(-self.data))
        out = tensor(y, (self,))

        # Calculate gradients for sigmoid()
        def _backward():
            self.grad += y * (1 - y) * out.grad
        out._backward = _backward

        return out
    
    def leaky_relu(self, slope=0.0):
        out_data = np.where(self.data >= 0, self.data, self.data * slope)
        out = tensor(out_data, (self,))

        # Calculate gradients for a leaky_relu()
        def _backward():
            self.grad += np.where(self.data >= 0, out.grad, out.grad * slope)
        out._backward = _backward

        return out 
    
    def exp(self):
        x = self.data
        out = tensor(np.exp(x), (self,))

        # Calculate gradients for exp()
        def _backward():
            self.grad = out.data * out.grad
        out._backward = _backward

        return out

    def backward(self):
        self.topo = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                self.topo.append(node)
        build_topo(self)

        self.grad = np.ones(self.shape) if self.shape else 1.0
        for node in reversed(self.topo):
            node._backward()

    def step(self, lr):
        for node in reversed(self.topo):
            if node.requires_grad:
                node.data -= lr * node.grad
            node.grad = 0.0

    def sum(self, axis=None, keepdims=False):
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = tensor(out_data, (self,))

        # Calculate gradients for sum()
        def _backward():
            self.grad += np.broadcast_to(out.grad, self.shape)
        out._backward = _backward

        return out

class randtensor(tensor):
    def __init__(self, shape, requires_grad=False):
        super().__init__(np.random.rand(*shape), requires_grad=requires_grad)

class zeros(tensor):
    def __init__(self, shape, requires_grad=False):
        super().__init__(np.zeros(shape), requires_grad=requires_grad)

class onehot(tensor):
    def __init__(self, shape, idx, requires_grad=False):
        array = np.zeros((1, shape))
        array[:, idx] = 1
        super().__init__(array, requires_grad=requires_grad)

class Linear_layer():
    def __init__(self, input_shape, output_shape):
        self.w = randtensor((input_shape, output_shape), requires_grad=True) * 0.01
        self.b = zeros((1, output_shape), requires_grad=True)

    def __call__(self, x):
        out = x @ self.w + self.b
        return out
    
class Sequential():
    def __init__(self, *args):
        self.layers = args
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
def BCELoss(logits, y):
    assert logits.shape == y.shape, "Shapes of logits and y must match"
    n = logits.shape[0]
    loss = -(1 / n) * (y * logits.log() + (1. - y) * (1. - logits).log()).sum()
    return loss

def MSELoss(logits, y):
    assert logits.shape == y.shape, "Shapes of logits and y must match"
    n = logits.shape[0]
    loss = (1 / n) * ((logits - y)**2).sum()
    return loss

def CrossEntropyLoss(logits, y):
    assert logits.shape == y.shape, "Shapes of logits and y must match"
    n = logits.shape[0]
    loss = -(1 / n) * (y * logits.log()).sum()
    return loss
    

def leaky_relu(x, slope=0.0):
    return x.leaky_relu(slope)

def sigmoid(x):
    return x.sigmoid()

def tanh(x):
    return x.tanh()

def softmax(x):
    y = x.exp()
    z = y / y.sum(axis=-1)
    return z
