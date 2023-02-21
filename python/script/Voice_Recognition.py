import wave 
import pyaudio 
from scipy.io import wavfile
import matplotlib.pyplot as plt 
import numpy as np 
import noisereduce as nr
import librosa 
import librosa.display
import torch
from torch import nn
import time
from torchvision import transforms
import struct 
import copy
import sys
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import threading
import csv
import socket

class Voice_Base(object):
    
    def __init__(self, path=[]):
        self.pa = pyaudio.PyAudio()
        self.path = path
    
    def audioread(self, return_nbits=True, formater='sample'):
        '''
                函数功能:读取语音文件,返回语音数据data,采样率fs,数据位数bits
        Input Param:
            formater:获取数据的格式,为sample时,数据为float32的[-1,1],
                     否则为文件本身的数据格式指定formater为任意非sample字符串,则返回原始数据
        Return:
            data: 语音数据
            fs: 采样率
            bits: 数据位数
        '''
        if return_nbits == True:
            f = wave.open(self.path)
            fs, data= wavfile.read(self.path)
            bit_depth=f.getsampwidth() * 8
        else:
            fs, data = wavfile.read(self.path)

        if formater == 'sample':
            data = data/(2**(bit_depth-1))
        if return_nbits:
            return data, fs, bit_depth 
        else:
            return data, fs
    

    def soundplot(self, data=[], samplerate=16000, size=(14,5)):
        """
        函数功能:将语音数据/或读取语音数据并绘制成图
        Input Param:
            data: 语音数据
            samplerate: 采样率
            size: 绘图窗口大小
        Return:
            None
        """
        if len(data) == 0:
            data, fs, _= self.audioread()
        plt.figure(figsize=size)
        x = [i/ samplerate for i in range(len(data))]
        plt.plot(x,data)
        plt.xlim([0, len(data) / samplerate])
        plt.xlabel('time/sec')
        plt.show()


    def soundadd(self, data1, data2):
        """
        函数功能: 将两个语音信号序列相加,若长短不一,在短的序列后端补零
        Input Param: 
            data1: 语音数据序列1
            data2: 语音数据序列2
        Return: 

        """
        if len(data1) < len(data2):
            temp = np.zeros([len(data2)])
            for i in range(len(data1)):
                temp[i] += data1[i]
            return temp + data2 

        elif len(data1) > len(data2):
            temp = np.zeros([len(data1)])
            for i in range(len(data2)):
                temp[i] += data2[i]
            return temp + data1 
        else:
            return data1 + data2


    def SPL_calculate(self, data, fs, frameLen):
        """
        函数功能: 根据数学公式计算单个声压值 $ y=\sqrt( \sum_{i=1} ^ Nx^2(i) ) $
        Input Param:
            data: 输入数据
            fs: 采样率
            frameLen: 计算声压的时间长度(ms单位)
        Return:
            单个声压数值
        """
        l = len(data)
        M = frameLen * fs / 1000
        if not l == M:
            exit('输入信号长度与所定义帧长不等!请检查!')
        # 计算有效声压
        pp = 0
        for i in range(int(M)):
            pp += (data[i] * data[i])
        pa = np.sqrt(pp / M)
        p0 = 2e-5
        spl = 20 * np.log10(pa / p0)

        return spl
    
    
    def SPL(self, data, fs, framLen=100, isplot=True):
        """
        计算声压曲线    
        Input Param:
            data: 语音信号数据
            fs: 采样率
            frameLen: 计算声压的时间长度(ms单位)
            isplot: 是否绘图，默认是
        Return: 
            返回声压列表spls
        """

        length = len(data)
        M = fs * framLen // 1000
        m = length % M 
        
        if not m < M // 2:
            # 最后一帧长度不小于M的一半
            data = np.hstack((data, np.zeros(M - m)))
        else:
            # 最后一帧长度小于M的一半
            data = data[:M * (length // M)]
        
        spls = np.zeros(len(data) // M)

        for i in range(len(data) // M - 1):
            s = data[i * M : (i+1) * M]
            spls[i] = self.SPL_calculate(s, fs, framLen)
        
        if isplot:
            plt.subplot(211)
            plt.plot(data)
            plt.subplot(212)
            plt.step([i for i in range(len(spls))], spls)
            plt.show()
        
        return spls 
    

    def enframe(self, x, win, inc=None):
        nx = len(x)
        if isinstance(win, list) or isinstance(win, np.ndarray):
            nwin = len(win)
            nlen = nwin  # 帧长=窗长
        elif isinstance(win, int):
            nwin = 1
            nlen = win  # 设置为帧长
        if inc is None:
            inc = nlen
        nf = (nx - nlen + inc) // inc
        frameout = np.zeros((nf, nlen))
        indf = np.multiply(inc, np.array([i for i in range(nf)]))
        for i in range(nf):
            frameout[i, :] = x[indf[i]:indf[i] + nlen]
        if isinstance(win, list) or isinstance(win, np.ndarray):
            frameout = np.multiply(frameout, np.array(win))
        return frameout

    # 矩形窗
    def reg_window(self, N):
        return np.ones(N)

    # 汉宁窗
    def hanning_window(self, N):
        nn = [i for i in range(N)]
        return 0.5 * (1 - np.cos(np.multiply(nn, 2 * np.pi) / (N - 1)))

    # 海明窗口
    def hamming_window(self, N):
        nn = [i for i in range(N)]
        return 0.54 - 0.46 * np.cos(np.multiply(nn, 2 * np.pi) / (N - 1))

    
    def STAc(self, x):
        """
        计算短时相关函数
        :param x:
        :return:
        """
        para = np.zeros(x.shape)
        fn = x.shape[1]
        for i in range(fn):
            R = np.correlate(x[:, i], x[:, i], 'valid')
            para[:, i] = R
        return para


    def STEn(self, x, win, inc):
        """
        计算短时能量函数
        :param x:
        :param win:
        :param inc:
        :return:
        """
        X = self.enframe(x, win, inc)
        s = np.multiply(X, X)
        return np.sum(s, axis=1)#-np.mean(s,axis=1)


    def STMn(self, x, win, inc):
        """
        计算短时平均幅度计算函数
        :param x:
        :param win:
        :param inc:
        :return:
        """
        X = self.enframe(x, win, inc)
        s = np.abs(X)
        return np.mean(s, axis=1)


    def STZcr(self, x, win, inc, delta=0):
        """
        计算短时过零率
        :param x:
        :param win:
        :param inc:
        :return:
        """
        absx = np.abs(x)
        x = np.where(absx < delta, 0, x)
        X = self.enframe(x, win, inc)
        X1 = X[:, :-1]
        X2 = X[:, 1:]
        s = np.multiply(X1, X2)
        sgn = np.where(s < 0, 1, 0)
        return np.sum(sgn, axis=1)#-np.mean(sgn, axis=1)


    def STAmdf(self, X):
        """
        计算短时幅度差，好像有点问题
        :param X:
        :return:
        """
        # para = np.zeros(X.shape)
        fn = X.shape[1]
        wlen = X.shape[0]
        para = np.zeros((wlen, wlen))
        for i in range(fn):
            u = X[:, i]
            for k in range(wlen):
                en = len(u)
                para[k, :] = np.sum(np.abs(u[k:] - u[:en - k]))
        return para


    def FrameTimeC(self, frameNum, frameLen, inc, fs):
        ll = np.array([i for i in range(frameNum)])
        return ((ll - 1) * inc + frameLen / 2) / fs


    def findSegment(self, express):
        """
        分割成语音片段
        :param express:
        :return:
        """
        if express[0] == 0:
            voiceIndex = np.where(express)
        else:
            voiceIndex = express
        d_voice = np.where(np.diff(voiceIndex) > 1)[0]
        voiceseg = {}
        if len(d_voice) > 0:
            for i in range(len(d_voice) + 1):
                seg = {}
                if i == 0:
                    st = voiceIndex[0]
                    en = voiceIndex[d_voice[i]]
                elif i == len(d_voice):
                    st = voiceIndex[d_voice[i - 1]+1]
                    en = voiceIndex[-1]
                else:
                    st = voiceIndex[d_voice[i - 1]+1]
                    en = voiceIndex[d_voice[i]]
                seg['start'] = st
                seg['end'] = en
                seg['duration'] = en - st + 1
                voiceseg[i] = seg
        return voiceseg

def findSegment(express):
        #"""
        #分割成语音段
        #:param express:
        #:return:
        #"""
        if express[0] == 0:
            voiceIndex = np.where(express)
        else:
            voiceIndex = express
        d_voice = np.where(np.diff(voiceIndex) > 1)[0]
        voiceseg = {}
        if len(d_voice) > 0:
            for i in range(len(d_voice) + 1):
                seg = {}
                if i == 0:
                    st = voiceIndex[0]
                    en = voiceIndex[d_voice[i]]
                elif i == len(d_voice):
                    st = voiceIndex[d_voice[i - 1] + 1]
                    en = voiceIndex[-1]
                else:
                    st = voiceIndex[d_voice[i - 1] + 1]
                    en = voiceIndex[d_voice[i]]
                seg['start'] = st
                seg['end'] = en
                seg['duration'] = en - st + 1
                voiceseg[i] = seg
        return voiceseg


def enframe(x, win, inc=None):
        nx = len(x)
        if isinstance(win, list) or isinstance(win, np.ndarray):
            nwin = len(win)
            nlen = nwin  # 帧长=窗长
        elif isinstance(win, int):
            nwin = 1
            nlen = win  # 设置为帧长
        if inc is None:
            inc = nlen
        nf = (nx - nlen + inc) // inc
        frameout = np.zeros((nf, nlen))
        indf = np.multiply(inc, np.array([i for i in range(nf)]))
        for i in range(nf):
            frameout[i, :] = x[indf[i]:indf[i] + nlen]
        if isinstance(win, list) or isinstance(win, np.ndarray):
            frameout = np.multiply(frameout, np.array(win))
        return frameout


def FrameTimeC(frameNum, frameLen, inc, fs):
        ll = np.array([i for i in range(frameNum)])
        return ((ll - 1) * inc + frameLen / 2) / fs


def vad_specEN(wnd,inc,fs,NIS,thr1,thr2,data,AU=None):
        import matplotlib.pyplot as plt
        from scipy.signal import medfilt
        if AU is None:
            x =enframe(data, wnd, inc)
        else:
            x =AU.enframe(data, wnd, inc)
        X = np.abs(np.fft.fft(x, axis=1))
        if len(wnd) == 1:
            wlen = wnd
        else:
            wlen = len(wnd)
        df = fs / wlen
        fx1 = int(250 // df + 1)  # 250Hz位置
        fx2 = int(3500 // df + 1)  # 500Hz位置
        km = wlen // 8
        K = 0.5
        E = np.zeros((X.shape[0], wlen // 2))
        E[:, fx1 + 1:fx2 - 1] = X[:, fx1 + 1:fx2 - 1]
        E = np.multiply(E, E)
        Esum = np.sum(E, axis=1, keepdims=True)
        P1 = np.divide(E, Esum)
        E = np.where(P1 >= 0.9, 0, E)
        Eb0 = E[:, 0::4]
        Eb1 = E[:, 1::4]
        Eb2 = E[:, 2::4]
        Eb3 = E[:, 3::4]
        Eb = Eb0 + Eb1 + Eb2 + Eb3
        prob = np.divide(Eb + K, np.sum(Eb + K, axis=1, keepdims=True))
        Hb = -np.sum(np.multiply(prob, np.log10(prob + 1e-10)), axis=1)
        for i in range(10):
            Hb = medfilt(Hb, 5)
        Me = np.mean(Hb)
        eth = np.mean(Hb[:NIS])
        Det = eth - Me
        T1 = thr1 * Det + Me
        T2 = thr2 * Det + Me
        voiceseg, vsl, SF, NF = vad_revr(Hb, T1, T2)
        return voiceseg, vsl, SF, NF, Hb

def vad_revr(dst1, T1, T2):
        """
        端点检测反向比较函数
        :param dst1:
        :param T1:
        :param T2:
        :return:
        """
        fn = len(dst1)
        maxsilence = 8
        minlen = 5
        status = 0
        count = np.zeros(fn)
        silence = np.zeros(fn)
        xn = 0
        x1 = np.zeros(fn)
        x2 = np.zeros(fn)
        for n in range(1, fn):
            if status == 0 or status == 1:
                if dst1[n] < T2:
                    x1[xn] = max(1, n - count[xn] - 1)
                    status = 2
                    silence[xn] = 0
                    count[xn] += 1
                elif dst1[n] < T1:
                    status = 1
                    count[xn] += 1
                else:
                    status = 0
                    count[xn] = 0
                    x1[xn] = 0
                    x2[xn] = 0
            if status == 2:
                if dst1[n] < T1:
                    count[xn] += 1
                else:
                    silence[xn] += 1
                    if silence[xn] < maxsilence:
                        count[xn] += 1
                    elif count[xn] < minlen:
                        status = 0
                        silence[xn] = 0
                        count[xn] = 0
                    else:
                        status = 3
                        x2[xn] = x1[xn] + count[xn]
            if status == 3:
                status = 0
                xn += 1
                count[xn] = 0
                silence[xn] = 0
                x1[xn] = 0
                x2[xn] = 0
        el = len(x1[:xn])
        if x1[el - 1] == 0:
            el -= 1
        if x2[el - 1] == 0:
            #print('Error: Not find endding point!\n')
            x2[el] = fn
        SF = np.zeros(fn)
        NF = np.ones(fn)
        for i in range(el):
            SF[int(x1[i]):int(x2[i])] = 1
            NF[int(x1[i]):int(x2[i])] = 0
        voiceseg = findSegment(np.where(SF == 1)[0])
        vsl = len(voiceseg.keys())
        return voiceseg, vsl, SF, NF


def getBaseMcff(file_path):
    with open(file_path,mode='r',encoding='utf-8') as file_obj:
        r=int(file_obj.readline());
        
        c=int(file_obj.readline());
        print(r,c)
        basemfcc=np.zeros((r,c));
        for i in range (r):
            for j in range (c):
                basemfcc[i][j]=float(file_obj.readline())
        file_obj.close();
    return np.transpose(basemfcc)


def noise_reduce(voice_data,noise_data=[],sample_rate=16000):
    reduced_voice_data=[]
    if len(voice_data)==0:
        print("collect noise")
    elif len(voice_data)!=0 and len(noise_data)==0:
        reduced_voice_data=nr.reduce_noise(y=voice_data,sr=sample_rate)
    elif len(voice_data)!=0 and len(noise_data)!=0:
        reduced_voice_data=nr.reduce_noise(y=voice_data,y_noise=noise_data,sr=sample_rate)
    return reduced_voice_data


#获取语音时序数据
def get_wav_time_data(path):
    AU = Voice_Base(path=path)
    data_two, fs, n_bits= AU.audioread()
    data= data_two
    #去噪
    data=noise_reduce(voice_data=data)
    #数据归一化
    data=data-np.mean(data)
    data /= np.max(data)
    return data,fs,n_bits,AU

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=nn.LSTM(
            #语音段的时序数
            input_size=20,
            #隐藏层
            hidden_size=64,
            #两层 RNN layers
            num_layers=2,
            #input & output 会是以batch size 为第一维度的特征集 e.g.(batch,time_step,input_size)
            batch_first=True,
        )
        #输出层
        self.out=nn.Linear(64,5)
    def forward(self,x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size)
        c0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size)
        r_out,(h_n,h_c)=self.rnn(x,(h0,c0))
        out=self.out(r_out[:,-1,:])
        return out


##回调函数中需要开启的线程
def Thread_process_voise(in_data):
    data=copy.copy(in_data)
    #去噪
    data=noise_reduce(voice_data=data)
    #数据归一化
    data=data-np.mean(data)
    data /= np.max(data)
    N=len(data)
    # ###取后五秒最新数据
    global Voise_Sequence_index
    # if(N>16000*5):
    #     data=data[-16000*5:]
    IS = 0.25
    wlen = 200
    inc = 200
    fs=RATE
    wnd = np.hamming(wlen)
    NIS = int((IS * fs - wlen) // inc + 1)
    thr1 = 0.99
    thr2 = 0.96
    #端点检测，提取音频段
    ########
    try:
        voiceseg, vsl, SF, NF, Enm =vad_specEN(wnd=wnd,inc=inc,fs=fs,NIS=NIS,thr1=thr1,thr2=thr2,data=data);
        fn = len(SF)
        frameTime =FrameTimeC(fn, wlen, inc, fs)
        for m in range(vsl):
            max=-999
            dis=voiceseg[m]['end']-voiceseg[m]['start']
            if (dis>25):
                data_seg_1 = data[ (int)(frameTime[voiceseg[m]['start']] * fs) : (int)(frameTime[voiceseg[m]['end']] *fs) ]
                #计算mfcc矩阵
                mfcc = librosa.feature.mfcc(y=data_seg_1, sr=fs)
                #将特征压入数据集中去
                r_,c_=mfcc.shape
                if c_>=10:
                    if(c_<20):
                        t=np.zeros((20,20-c_))
                        mfcc=np.column_stack((mfcc,t))
                    elif c_ >20:
                        mfcc=mfcc[:,:20]
                    mfcc=np.transpose(mfcc)
                    #获取转置的mfcc矩阵，每行为一个时序的20个频段的数据分布
                    mfcc=transform(mfcc).to(torch.float32)
                    #获取网络给出的类型返回
                    x=net(mfcc).detach().numpy()[0]
                    if(vsl-m<5):

                        print(m,x,"持续时间：",dis)
                    #获取可信度最大的值和其对应的下标
                    for i in range (5):
                        if x[i]>max:
                            max=x[i]
                            index=i
                    #可信度大于0就更新Voise_Sequence数组
                    if(max>0):
                        Voise_Sequence[m]=Voise_Class[index]+' confidence:'+str(max)
                    #更新Voise_Sequence_index
                    if(m>Voise_Sequence_index):
                        Voise_Sequence_index=m
                        #热身结束，向树莓派发送数据
                        if(Voise_Sequence_index>2):
                            pool.submit(send_message,Voise_Class[index],dis)
    except(IndexError):
        xxxxx=0
    
def send_message(msg,dis):
    
    
    if(dis<50):
        msg_tail=',S'
    else:
        msg_tail=',L'
    msg=msg+msg_tail
    try:
        client_socket.send(msg.encode('utf-8'))
        print("发送成功；")
    except :                           ##连接不成功，运行最初的ip
        print ('发送失败')
        return
    mySocket.send(msg.encode("utf-8"))


def updata_whole_data(data):
    dataList.append(copy.copy(data))


##pyaudio录制语音用的回调函数
def myCallback(
    in_data,      # 如果input=True，in_data就是录制的数据，否则为None
    frame_count,  # 帧的数量，表示本次回调要读取几帧的数据
    time_info,    # 一个包含时间信息的dict，略
    status_flags  #标记位
):
    pool.submit(updata_whole_data,in_data)
    stream_data=in_data
    stream_data=struct.unpack('<1024h',bytes(stream_data)) 
    global data
    global len_of_last_process_data
    global dis_of_process_data
    #数据合并
    data=data+(list)(stream_data)
    #前一秒不做处理，用作收集降噪数据
    if len(data)<20000:
        print("收集噪声中")
        return b"", pyaudio.paContinue
    
    #临时数据，处理用
    #不是每次都检查数据
    if len(data)-len_of_last_process_data>dis_of_process_data :      
        #提交线程处理最新的数据
        pool.submit(Thread_process_voise, data)
        #更新数据
        len_of_last_process_data=len(data)
    return b"", pyaudio.paContinue


def Check_Voise_Sequence():
    while True:
        if Voise_Sequence_index>-1:
            #打印信息
            #print(Voise_Sequence[0:Voise_Sequence_index-1])
            #更新txt
            ##TODO 修改路径
            path="C:/Users/gnyy/Desktop/Underwater_Superlimb-master/underwater_project/python/script/wav/Voise_Sequence/Voise_Sequence.csv"
            file=open(path,'w',newline='')
            csv_writer=csv.writer(file,dialect="excel")
            csv_writer.writerow(['class','confidence'])
            for i in range (Voise_Sequence_index+1):
                row=[]
                row.append(Voise_Sequence[i][0:2])
                row.append(Voise_Sequence[i][14:])
                csv_writer.writerow(row)
            file.close()
        print(Voise_Sequence_index)
        time.sleep(0.5)
        if(End_Flag==1):
            return

def send_zero():
    msg='0'
    while(True):
        time.sleep(0.5)
        if(End_Flag==1):
            break
        try:
            #client_socket.send(msg.encode('utf-8'))
            x=0
        except :                           ##连接不成功，运行最初的ip
            print ('发送失败')
            mySocket.send(msg.encode("utf-8"))
#######疯狂写注释摸鱼
######啊啊啊啊难顶


################整理一下思路，我的数据是要实时的，有一个整体的data，可以切出最新的五秒数据，
##获取实时语音数据流


##############参语音录制数设置

#初始化socket

#套接字接口
mySocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
mySocket.settimeout(50000)
#设置ip和端口
host = '10.13.228.137'
port = 7788
mySocket.bind((host,port))
mySocket.listen(10)
print("start")
client_socket, clientAddr = mySocket.accept()

#TODO 修改路径
rnnPath='C:/Users/gnyy/Desktop/Underwater_Superlimb-master/underwater_project/python/script/rnn2.pth'
recorsPath="C:/Users/gnyy/Desktop/Underwater_Superlimb-master/underwater_project/python/script/wav/ALL/1.wav"




#语音帧大小
CHUNK = 1024
#数据深度
FORMAT = pyaudio.paInt16
#语音通道数
CHANNELS = 1
#采样率
RATE = 16000

#########################前置数据设置
#Pyaudio初始化
p = pyaudio.PyAudio()


###声音类别
Voise_Class=['do','re','me','fa','so']

###数据处理用数组
data=[]

##返回语音文件用数组
dataList=[]

##数据转换用，主要用于将数据转换至tensor
transform=transforms.ToTensor()
#TODO 修改路径
rnnPath='C:/Users/gnyy/Desktop/Underwater_Superlimb-master/underwater_project/python/script/rnn2.pth'
##网络初始化
net = torch.load(rnnPath)

###上次处理数据的长度，用于与当前数据的长度比较
len_of_last_process_data=0

###当数据增长到一定程度时，才对后五秒数据进行处理
dis_of_process_data=5120


#初始化线程池，最大线程池设置为200，够用了吧，应该
pool = ThreadPoolExecutor(max_workers=200)
#pool2=ThreadPoolExecutor(max_workers=1)

#pool2.submit(send_zero)


#类型返回数据，下标表示第几个数据，该数组表示整个数据的类型序列，
Voise_Sequence=[]
for i in range(500):
    Voise_Sequence.append('-1')

##类型返回数据是冗余的，这是为了方式线程在更新数组时导致的越界问题，因此需要一个标记来标记当前有效语音段的最大数据
Voise_Sequence_index=-1


##语音结束标记，用于结束轮询线程
End_Flag=0
#打开录音流
stream =p.open(format=FORMAT,#数据深度
               channels=CHANNELS,#频道数
               rate=RATE,#采样率
               input=True,#输入流
               frames_per_buffer=CHUNK,#设置数据帧大小
               stream_callback=myCallback) #设置回调函数

print("开始")

#开启一个线程，轮询Voise_Sequence，每一秒更新一下数据


#开始录音
stream.start_stream()
t1 = time.time()

#pool.submit(Check_Voise_Sequence)
#用于录音是否结束，录音时长为60，time.sleep的目的时为了减少麦克风输入或者输出延迟的影响
while time.time() - t1 < 120:
    time.sleep(0.01)

End_Flag=1

#TODO 修改路径
recorsPath="C:/Users/gnyy/Desktop/Underwater_Superlimb-master/underwater_project/python/script/wav/ALL/1.wav"
#写文件
wav_data = b"".join(dataList)
with wave.open(recorsPath, "wb") as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(wav_data)





stream.stop_stream()
stream.close()
p.terminate()
#Check_Voise_Sequence()
print("结束了")
client_socket.close()



#################3
#类型返回数据，下标表示第几个数据，该数组表示整个数据的类型序列，


# Voise_Sequence=[]
# for i in range(500):
#     Voise_Sequence.append('-1')
# Voise_Class=['do','re','me','fa','so']

# Voise_Sequence_Num=[]
# Time=[]
# ##类型返回数据是冗余的，这是为了方式线程在更新数组时导致的越界问题，因此需要一个标记来标记当前有效语音段的最大数据
# Voise_Sequence_index=-1
# max_=-999
# index=-1


# #######
#TODO 修改路径
# path="C:/Users/gnyy/Desktop/Underwater_Superlimb-master/underwater_project/python/script/wav/ALL/2.wav"
# data,fs,n_bits,AU=get_wav_time_data(path)
# transform=transforms.ToTensor()
#TODO 修改路径
# net = torch.load('C:/Users/gnyy/Desktop/Underwater_Superlimb-master/underwater_project/python/script/rnn2.pth')


# IS = 0.25
# wlen = 200
# inc = 200
# N = len(data)
# wnd = np.hamming(wlen)
# time = [i / fs for i in range(N)]
# overlap = wlen - inc
# NIS = int((IS * fs - wlen) // inc + 1)
# thr1 = 0.99
# thr2 = 0.96
# #端点检测，提取音频段
# voiceseg, vsl, SF, NF, Enm =vad_specEN(AU=AU,wnd=wnd,inc=inc,fs=fs,NIS=NIS,thr1=thr1,thr2=thr2,data=data);
# fn = len(SF)
# frameTime = AU.FrameTimeC(fn, wlen, inc, fs)
# index=-1

# plt.figure(figsize=(18,10))

# plt.subplot(3, 1, 1)
# plt.plot(time, data)
# plt.ylabel("amplitude")
# plt.xlabel("time/s")
# plt.subplot(3, 1, 2)
# plt.ylabel("amplitude")
# plt.xlabel("time/s")
# plt.plot(frameTime, Enm)

# last_end=0

# #遍历选定音频下的所有声音，计算mfcc特征矩阵
# for m in range(vsl):


#     if(m==0):
#         for i in range(voiceseg[m]['end']+1):
#             Voise_Sequence_Num.append(0)
#     else:
#         for i in range(last_end,voiceseg[m]['end']+1):
#             Voise_Sequence_Num.append(index+1)
#     if(voiceseg[m]['end']-voiceseg[m]['start']>20):
#         plt.subplot(3, 1, 1)
#         plt.plot(frameTime[voiceseg[m]['start']], 0, '.k')
#         plt.plot(frameTime[voiceseg[m]['end']], 0, 'or')
#         plt.legend(['signal', 'start', 'end'])
#         plt.subplot(3, 1, 2)
#         plt.plot(frameTime[voiceseg[m]['start']], 0, '.k')
#         plt.plot(frameTime[voiceseg[m]['end']], 0, 'or')
#         plt.legend(['SE', 'start', 'end'])
#         plt.figure(figsize=(18,10))
#         data_seg_1 = data[ (int)(frameTime[voiceseg[m]['start']] * fs) : (int)(frameTime[voiceseg[m]['end']] *fs) ]
#         time_seg_1 = np.linspace(frameTime[voiceseg[m]['start']],frameTime[voiceseg[m]['end']], len(data_seg_1))
#         plt.plot(time_seg_1, data_seg_1)
#         #计算mfcc矩阵
#         tempmfcc = librosa.feature.mfcc(y=data_seg_1, sr=fs)
#         #将特征压入数据集中去
#         r_,c_=tempmfcc.shape
#         if c_>=15:
#             if(c_<20):
#                 t=np.zeros((20,20-c_))
#                 tempmfcc=np.column_stack((tempmfcc,t))
#             elif c_ >20:
#                 tempmfcc=tempmfcc[:,:20]
#             tempmfcc=np.transpose(tempmfcc)
#             #获取转置的mfcc矩阵，每行为一个时序的20个频段的数据分布
#             tempmfcc=transform(tempmfcc).to(torch.float32)
#             print(m,net(tempmfcc))
#             x=net(tempmfcc).detach().numpy()[0]
#             #获取可信度最大的值和其对应的下标
#             for i in range (5):
#                 if x[i]>max_:
#                     max_=x[i]
#                     index=i
#             #可信度大于0就更新Voise_Sequence数组
            
#             if(max_>0):
#                 Voise_Sequence[m]=Voise_Class[index]+' confidence:'+str(max_)
#                 #更新Voise_Sequence_index
#                 if(m+1>Voise_Sequence_index):
#                     Voise_Sequence_index=m+1
#             max_=-999

#             last_end=voiceseg[m]['end']+1
#             if(m==vsl-1):
#                 for i in range(len(frameTime)-len(Voise_Sequence_Num)):
#                     Voise_Sequence_Num.append(index+1)
#         plt.close()
# for i in range(Voise_Sequence_index):
#     print(Voise_Sequence[i])


# plt.subplot(3,1,3)
# plt.ylabel("class")
# plt.xlabel("time/s")
# plt.plot(frameTime,Voise_Sequence_Num,'g-')

# plt.show()