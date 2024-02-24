if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import os
import subprocess
import urllib.request
import numpy as np
from dezero import Variable

def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()
    
    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
    
    add_func(output.creator)
    txt += _dot_var(output, verbose)
    
    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)
            
            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g {\n' + txt + '}'

def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'
    
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)

def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)
    
    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))  # yはweakref
    return txt

def plot_dot_graph(output, verbose=True, to_file='graph.png', isCd=True):
    dot_graph = get_dot_graph(output, verbose)
    
    # ①dotデータをファイルに保存
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir):  # ~/.dezeroディレクトがない場合作成
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')
    
    with open(graph_path, 'w') as f:
        f.write(dot_graph)
    
    # ②dotコマンドを呼ぶ
    extension = os.path.splitext(to_file)[1][1:]  # 拡張子(png, pdfなど)
    if not isCd:
       to_file = '/Users/sora/Desktop/dezeroSA/dots/' + to_file 
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)

def sum_to(x, shape):
    """
    step40.pyに動作イメージあり
    xはsum_to前の値、shapeはsum_to後の値の形状
    """
    # 順伝播の出入力の次元数の差を計算
    ndim = len(shape)
    lead = x.ndim - ndim
    """step43で追加"""
    lead_axis = []
    exist_axis = []
    axis = []
    # if (lead > 0):
    x_shape_lst, shape_lst = list(x.shape), list(shape)
    next_idx = 0
    for i, sdm in enumerate(shape_lst):
        if (sdm == 1):
            continue
            
        j = next_idx 
        while j < len(x_shape_lst):
            if (sdm == x_shape_lst[j]):
                exist_axis.append(j)
                next_idx = j + 1
                break
            else:
                j += 1
    
    first_exist_axis = -1
    if (len(exist_axis) > 0):
        first_exist_axis = exist_axis[0]
    
    for i, sdm in enumerate(shape_lst):
        if (sdm == 1):
            if (i < first_exist_axis):
                axis.append(i)
            else:
                axis.append(i + lead)
    
    for i, _ in enumerate(x_shape_lst):
        if (i not in axis + exist_axis):
            lead_axis.append(i)

    axis = tuple(axis)
    lead_axis = tuple(lead_axis)
    """step43で追加""" 
    
    ## 追加された軸番号を作成（軸は先頭に追加されるため、range(lead)で追加された軸番号を取得可能）
    #lead_axis = tuple(range(lead))
    #lead_axis = tuple(range(ndim, ndim + lead))

    # """
    # ブロードキャストで要素数(sx)が１の次元が複製される。
    # #このとき、複製された次元の軸番号(i)は、追加された軸数(lead)分後ろになる。
    # #a: (0, 1), b: (2, 3, 4)の時、a + b: (0, 1, 2, 3, 4)
    # #sum(全ての軸)は、sum()と等しい（全ての要素の合計）。
    # #x: [[0, 1, 2], [3, 4, 5]]の時、 x.sum((0, 1)): 15、x.sum((0, 1), keepdims=True): [[15]]
    # """
    # 和を求める際に消去される軸番号を抽出
    #axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    #axis = tuple([i for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    
    """
    np.squeeze()に指定した軸を取り除きます。lead_axisが空()のときは、要素数が1の軸を全て消去します。
    leadが0の場合は、順伝播の入出力(逆伝播の出入力)の形状に差がないので軸を消去する必要がありません。
    """
    # 順伝播の入力の形状に整形
    if lead > 0:
        # 不要な軸を消去
        y = y.squeeze(lead_axis)
    return y

def reshape_sum_backward(gy, x_shape, axis, keepdims):
    ndim = len(x_shape)
    tupled_axis = axis
    # axisを指定しなかった場合
    if axis is None:
        tupled_axis = None
    # axisにスカラを指定した場合
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    # 順伝播時に消去された軸を復元
    # xがスカラでない or axisを指定した or keepdimsがFalseの場合
    if not (ndim == 0 or tupled_axis is None or keepdims):
        # 和をとった軸の番号を(前から数えた番号に変換して)抽出
        """
        和をとることにより一部の軸のみが消えています。ブロードキャストを行うには、消去された軸を要素数を1として復元させる必要があります。
        tupled_axisは和をとった軸の番号でした。軸は負の値を使って指定することもできます。-1は最後の軸を表し、-nは後からn番目の軸を表します。
        この例だと、軸の数(次元数)ndimは3ですね。つまり、-1は(0から数えるので)2番目の軸を表します。これは、-1 + ndimで求められます。
        同様に、-nは前から数えると-n + ndim番目の軸です。

        このことを考慮して、和をとった軸の番号を取得しています。前から数えた番号で軸を指定している場合はa >= 0なので、そのままactual_axisに追加します。
        後から数えた番号で軸を指定している場合はa < 0なので、ndimの値を加えてactual_axisに追加します。
        次に、現在の形状(順伝播の出力の形状)をリスト型に変換してshapeとして保存します。

        さらに、shapeに対してリストのメソッドinsert()を使って、和をとることによって消去した軸actual_axisを追加してその軸の要素数を1とします。
        insert()は、第1引数に指定したインデックスに、第2引数に指定した要素を追加します。

        以上で調整後の形状shapeが得られました。
        
        例1 x.shape: (2, 3) -> y = x.sum(axis=0) -> y.shape: (3,)の時、actual_axis: [0] -> [3].insert(0, 1) -> shape: [1, 3]
        例2 x.shape: (2, 3) -> y = x.sum(axis=1) -> y.shape: (2,)の時、actual_axis: [0] -> [2].insert(1, 1) -> shape: [2, 1] 
        """
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        # 順伝播の出力の形状をリスト型で保存
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    # 調整が不要な変形の場合
    else:
        # 現在の形状を保存
        shape = gy.shape

    gy = gy.reshape(shape)
    return gy

def logsumexp(x, axis=1):
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    np.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    np.log(s, out=s)
    m += s
    return m

def show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0: p = 100.0
    if i >= 30: i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end='')

cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')

def get_file(url, file_name=None):
    """Download a file from the `url` if it is not in the cache.

    The file at the `url` is downloaded to the `~/.dezero`.

    Args:
        url (str): URL of the file.
        file_name (str): Name of the file. It `None` is specified the original
            file name is used.

    Returns:
        str: Absolute path to the saved file.
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path

def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError

def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1

# (inputs_size + pad * 2 - kernel_size) // stride + 1 = outputs_sizeの式より、
# inputs_sizeについて解くと下記になる。（s:stride, size:outputs_size, k:kernel_size, p:pad）
def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2 * p