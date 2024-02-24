import numpy as np
from dezero.core import Function, as_variable
from dezero.utils import pair, get_conv_outsize, get_deconv_outsize
from dezero.functions import linear, broadcast_to

class Conv2d(Function):
    def __init__(self, stride=1, pad=0):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
    
    def forward(self, x, W, b):
        # Wは４次元データ（OC, C, KH, KW）
        KH, KW = W.shape[2:]
        # colは６次元データ（N, C, KH, KW, OH, OW）
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False)
        
        # colの1,2,3軸(C,KH,KW)とWの1,2,3軸(C,KH,KW)でtensor積を計算
        # yは４次元データ（N, OH, OW, OC)
        y = np.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b
        # yの3番目の軸を１番目に移動（N, OH, OW, OC）->（N, OC, OH, OW）　＜出力マップのサイズ＞
        y = np.rollaxis(y, 3, 1)
        return y
    
    def backward(self, gy):
        # gyは４次元データ（N, OC, OH, OW）　＜出力マップのサイズ＞
        x, W, b = self.inputs
        # gxは4次元データ（N, C, H, W）　＜入力データのサイズ＞
        gx = deconv2d(gy, W, b=None, stride=self.stride, pad=self.pad, outsize=(x.shape[2], x.shape[3]))
        #  gwは４次元データ（OC, C, KH, KW）　＜カーネルのサイズ＞
        gW = Conv2DGradW(self)(x, gy)
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb

def conv2d(x, W, b=None, stride=1, pad=0):
    return Conv2d(stride, pad)(x, W, b)

class Deconv2d(Function):
    # 参照:https://techblog.nhn-techorus.com/archives/12879
    def __init__(self, stride=1, pad=0, outsize=None):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.outsize = outsize
    
    def forward(self, x, W, b):
        Weight = W
        SH, SW = self.stride
        PH, PW = self.pad
        # Cがフィルタ数、OCがフィルタのチャンネル数している。
        C, OC, KH, KW = Weight.shape
        # H, Wは出力マップのサイズ（OH, OW）、Cは出力マップのチャンネル数（OC）
        N, C, H, W = x.shape
        if self.outsize is None:
            out_h = get_deconv_outsize(H, KH, SH, PH)
            out_w = get_deconv_outsize(W, KW, SW, PW)
        else:
            out_h, out_w = pair(self.outsize)
            
        # out_hとout_wは元画像の高さと幅、OCはチャンネル数だから、img_shapeは元画像のサイズになる。
        img_shape = (N, OC, out_h, out_w)
        
        # Weightの0軸(C)とxの1軸(C)でtensor積を計算
        # gcolは４次元データ（OC, KH, KW, N, H, W)
        gcol = np.tensordot(Weight, x, (0, 1))
        # gcolの3番目の軸を0番目に移動（OC, KH, KW, N, H, W）->（N, OC, KH, KW, H, W）
        gcol = np.rollaxis(gcol, 3)
        
        # yは４次元データ（N, OC, out_h, out_w） ※実際は（N, C, H, W）　＜入力データのサイズ＞
        y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad, to_matrix=False)
        
        if b is not None:
            self.no_bias = True
            y += b.reshape((1, b.size, 1, 1))
        return y
    
    def backward(self, gy):
        # gyは４次元データ（N, C, H, W）　＜入力データのサイズ＞
        x, W, b = self.inputs
        
        # gxは４次元データ（N, OC, OH, OW）　＜出力マップのサイズ＞
        gx = conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
        #  gwは４次元データ（OC, C, KH, KW）　＜カーネルのサイズ＞
        f = Conv2DGradW(self)
        gW = f(gy, x)
        gb = None
        if b.data is not None:
            gb = gy.sum(zxis=(0, 2, 3))
        return gx, gW, gb

def deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):
    return Deconv2d(stride, pad, outsize)(x, W, b)

class Conv2DGradW(Function):
    def __init__(self, conv2d):
        # Wは４次元データ（OC, C, KH, KW） 
        W = conv2d.inputs[1]
        kh, kw = W.shape[2:]
        self.kernel_size = (kh, kw)
        self.stride = conv2d.stride
        self.pad = conv2d.pad
    
    def forward(self, x, gy):
        # xは４次元データ（N, C, H, W）　＜入力データのサイズ＞
        # gyは４次元データ（N, OC, OH, OW）　＜出力データのサイズ＞
        # colは６次元データ（N, C, KH, KW, OH, OW）
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        # gwは４次元データ（OC, C, KH, KW）　＜カーネルのサイズ＞
        gW = np.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
        return gW
    
    def backward(self, gys):
        x, gy = self.inputs
        gW, = self.outputs
        
        xh, xw = x.shape[2:]
        # gxは４次元データ（N, C, H, W）　＜入力データのサイズ＞
        gx = deconv2d(gy, gW, stride=self.stride, pad=self.pad, outsize=(xh, xw))
        # ggyは４次元データ（N, OC, OH, OW）　＜出力データのサイズ＞
        ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
        return gx, ggy

def conv2d_simple(x, W, b=None, stride=1, pad=0):
    x, W = as_variable(x), as_variable(W)
    
    Weight = W
    N, C, H, W = x.shape
    OC, C, KH, KW = Weight.shape
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)
    
    col = im2col(x, (KH, KW), stride, pad, to_matrix=True)
    Weight = Weight.reshape(OC, -1).transpose()
    t = linear(col, Weight, b)
    y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)
    return y


class Pooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        # xは４次元データ（N, C, H, W）
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        """
        OH,OW行列内の各要素ごとに、KH*KW個の中で最大値の値を持つインデックス(0〜KH*KW-1)を返す。
        OH*OW*C*Nの要素ごとにKH*KW個の中から最大値を求めているから、pooling_simpleとやっていることは同じ。（より効率が良い）
        """
        # self.indexesは４次元データ（N, C, OH, OW）
        self.indexes = col.argmax(axis=2)
        # yは４次元データ（N, C, OH, OW）
        y = col.max(axis=2)
        return y
    
    def backward(self, gy):
        # gyは４次元データ（N, C, OH, OW） 
        return Pooling2DGrad(self)(gy)

def Pooling2DGrad(Function):
    def __init__(self, mpool2d):
        self.mpool2d = mpool2d
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        # input_shapeは４次元データ（N, C, H, W） 　＜入力データのサイズ＞
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes
    
    def forward(self, gy):
        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)
        
        gcol = np.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)
        
        """
        indexes.size=N*C*OH*OWより、np.arangeは0〜N*C*OH*OW*KH*W-1までの値を間隔KH*KWで取得。
        また、indexes.ravel()は、要素を１次元に展開する。
        よって、1次元に展開されたOH,OW行列内の各要素ごとに求めたKH*KW個の中で最大値の値を持つインデックス
        に対して、0〜N*C*OH*OW*KH*W-1までの値を間隔KH*KWで取得した値（要素数は同じ）を足すことで、
        0〜N*C*OH*OW*KH*W-1という展開された要素の中での最大値を持つインデックスに変換している。
        """
        indexes = (self.indexes.ravel() + np.arange(0, self.indexes.size * KH * KW, KH * KW))
        
        # gcolのindexesの示す場所(N*C*OH*OW個)に、展開されたgyの要素(N*C*OH*OW個)を順番に格納していく。
        gcol[indexes] = gy.ravel()
        # gcolを6次元データ（N, C, OH, OW, KH, KW）にする
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        # gcolの２番目（OH）と４番目（KH）の軸を入れ替える -> （N, C, KH, OW, OH, KW）
        gcol = np.swapaxes(gcol, 2, 4)
       # gcolの３番目（OH）と５番目（KH）の軸を入れ替える -> （N, C, KH, KW, OH, OW） 
        gcol = np.swapaxes(gcol, 3, 5)
        # gxは4次元データ（N, C, H, W）  　＜入力データのサイズ＞
        gx = col2im_array(gcol, (N, C, H, W), self.kernel_size, self.stride, self.pad, to_matrix=False)
        
        return gx
    
    def backward(self, ggx):
        # ggxは4次元データ（N, C, H, W） 　＜入力データのサイズ＞
        f = Pooling2DWithIndexes(self.mpool2d)
        return f(ggx)

class Pooling2DWithIndexes(Function):
    def __init__(self, mpool2d):
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        # input_shapeは４次元データ（N, C, H, W） 　＜入力データのサイズ＞
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes
    
    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        # colは5次元データ（N, C, KH*KW, OH, OW）
        col = col.reshape(N, C, KH * KW, OH, OW)
        # colは(N, C, OH, OW , KH*KW) -> (N*C*OH*OW, KH*KW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
        # indexesは1次元データ（N*C*OH*OW）
        indexes = self.indexes.ravel()
        # colの要素にindexesの対応する要素を格納していく。（indexesの要素を格納する際に、その要素はKH*KW個にブロードキャストされる）
        col = col[np.arange(len(indexes)), indexes]
        # (N, C, OH, OW)は＜出力データのサイズ＞
        return col.reshape(N, C, OH, OW)

def pooling(x, kernel_size, stride=1, pad=0):
    return Pooling(kernel_size, stride, pad)(x)

def pooling_simple(x, kernel_size, stride=1, pad=0):
    x = as_variable(x)
    
    N, C, H, W = x.shape
    KH, KW = pair(kernel_size)
    PH, PW = pair(pad)
    SH, SW = pair(stride)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)
    
    col = im2col(x, kernel_size, stride, pad, to_matrix=True)
    col = col.reshape(-1, KH * KW)
    y = col.max(axis=1)
    y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
    return y


class Im2col(Function):
    def __init__(self, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix
    
    def forward(self, x):
        self.input_shape = x.shape
        y = im2col_array(x, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return y
    
    def backward(self, gy):
        gx = col2im(gy, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return gx

def im2col(x, kernel_size, stride=1, pad=0, to_matrix=True):
    y = Im2col(kernel_size, stride, pad, to_matrix)(x)
    return y

class Col2im(Function):
    def __init__(self, input_shape, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        y = col2im_array(x, self.input_shape, self.kernel_size, self.stride,
                         self.pad, self.to_matrix)
        return y

    def backward(self, gy):
        gx = im2col(gy, self.kernel_size, self.stride, self.pad,
                    self.to_matrix)
        return gx

def col2im(x, input_shape, kernel_size, stride=1, pad=0, to_matrix=True):
    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)

def im2col_array(img, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)
    
    # N, C方向：パディングしない。
    # H方向   ：上下PHずつパディング。右側はstrideが２以上の場合を考慮して、最後のはみ出る分を追加しておく。）
    # W方向   ：左右PWずつパディング。下側はstrideが２以上の場合を考慮して、最後のはみ出る分を追加しておく。）
    img = np.pad(img, ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
                 mode='constant', constant_values=(0,))
    # ６次元データ（N, C, KH, KW, OH, OW）を作成
    col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)
    
    """
    フィルタの高さ分(0～filter_h-1)でループ。
    このとき、y_maxに設定される値の意味は、出力データの高さ分をimgから取得するには、
    imgの高さの次元のy番目から出力データの高さ分のストライドを進めた位置まで、指定のストライドでデータを取得することを表す。
        例：imgの高さ次元の大きさ11,フィルタの高さ次元の大きさ3,ストライド2,出力データの高さ次元の大きさ5のとき、j:j_lim:SHは、
          ①j=0のとき：j_lim=10より、imgから0, 2, 4, 6, 8の５つの高さ次元のデータ取得
          ②j=1のとき：j_lim=11より、imgから1, 3, 5, 7, 9の５つの高さ次元のデータ取得
          ③j=2のとき：j_lim=12より、imgから2, 4, 6, 8, 10の５つの高さ次元のデータ取得
          　※j=2のとき、(2-4), (4-6), (6-8), (8-10), (10-12)で高さ次元をスライシングするため、
             PH + SH - 1だけ、大きくする。今回の場合は、0 + 2 - 1 = 1だけimgの高さ次元の大きさが大きくなっている(11 -> 12)。
    xについても同様の処理が行われる。
    これによって、imgからの４次元データ（N個の各データのCチャネル上のスライシングしたyとスライシングしたxの行列）を、
    対応するcolのN個の各データのCチャネル上のy, x番目の位置にそれぞれコピーする。※NxC個の分の処理を同時に行っている。
    """
    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OH
            col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]
    
    """
    colを(N, OH, OW, C, KH, KW)の軸順の６次元データにする。
    その後、(N * OH * OW, C * KH * KW)の２次元データにする。
    """
    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))
    
    return col

def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    """
    colを(N, OH, OW, C, KH, KW)の軸順の６次元データにする。
    その後、(N, C, KH, KW, OH, OW)の６次元データにする。
    """ 
    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    # imgのHとW次元の大きさをpad分大きくする。（また、strideが２以上の場合stride-1分HとWを大きくする。）
    img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1), dtype=col.dtype)
    
    """
    im2colと逆で、im2colでスライシングして格納した位置のcolの値をimgの元の位置に加算する。
    ※加算する理由は、im2col時のスライシングで重複する箇所があるため。（右記のgif参照:https://qiita.com/kuroitu/items/7877c002e192955c7857）
    そして、最終的にpad部分を消去したimgを返す。
    """
    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
            
    return img[:, :, PH:H + PH, PW:W + PW]
