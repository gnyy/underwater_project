'''
Date: 2022-11-14 16:28:57
LastEditors: Guo Yuqin,12032421@mail.sustech.edu.cn
LastEditTime: 2022-11-24 01:44:45
FilePath: /script/Voice_Base.py
'''

import wave 
import pyaudio 
from scipy.io import wavfile
import matplotlib.pyplot as plt 
import numpy as np 
import noisereduce as nr
import os
import librosa 
import librosa.display
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
#计算曼哈顿距离的匿名函数
manhattan_distance=lambda x,y:np.sum(np.abs(x-y))
#引入第三方dtw包直接计算
from dtw import dtw
# plt.rcParams['font.family'] = ['sans-serif']
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

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

def vad_specEN(AU,wnd,inc,fs,NIS,thr1,thr2,data):
        import matplotlib.pyplot as plt
        from scipy.signal import medfilt
        x = AU.enframe(data, wnd, inc)
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
            print('Error: Not find endding point!\n')
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
        print("语音数据为空")
    elif len(voice_data)!=0 and len(noise_data)==0:
        reduced_voice_data=nr.reduce_noise(y=voice_data,sr=sample_rate)
    elif len(voice_data)!=0 and len(noise_data)!=0:
        reduced_voice_data=nr.reduce_noise(y=voice_data,y_noise=noise_data,sr=sample_rate)
    return reduced_voice_data


def listdir(path):  # 传入存储的list
    list_name=[]
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if not os.path.isdir(file_path):
            list_name.append(file_path)
    return list_name


def create_wav_file(path,data,fs):

    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = fs

    p = pyaudio.PyAudio()
    wf = wave.open(path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(data))
    wf.close()

#获取wav数据及其标签组成的数据
#方便统一快速地制作训练集和测试集
#i:数据集起点下标，如do_i.wav
#j:数据集终点下标，如do_j.wav
def get_wav_data(i,j):

    #负责记录整个数据集的矩阵
    whole_data=[]
    wave_data_class=["do","re","mi","fa","so"]
    index=0
    #c:class
    for c in wave_data_class:
        #打开当前语音种类对应的文件夹，预备数据
        path="C:/Users/gnyy/Desktop/Underwater_Superlimb-master/wav_data/"+c
        #获取语音数据列表
        dir_list=listdir(path=path)
        #遍历语音数据，挑选指定下标范围的数据
        for n in range(len(dir_list)):
            if n>=i-1 and n<j:
                data,fs,n_bits,AU=get_wav_time_data(dir_list[n])
                IS = 0.25
                wlen = 200
                inc = 200
                N = len(data)
                wnd = np.hamming(wlen)
                time = [i / fs for i in range(N)]
                overlap = wlen - inc
                NIS = int((IS * fs - wlen) // inc + 1)
                thr1 = 0.99
                thr2 = 0.96
                #端点检测，提取音频段
                voiceseg, vsl, SF, NF, Enm =vad_specEN(AU=AU,wnd=wnd,inc=inc,fs=fs,NIS=NIS,thr1=thr1,thr2=thr2,data=data);
                fn = len(SF)
                frameTime = AU.FrameTimeC(fn, wlen, inc, fs)
                #遍历选定音频下的所有声音，计算mfcc特征矩阵
                for m in range(vsl):
                    if m>=1:
                        data_seg_1 = data[ (int)(frameTime[voiceseg[m]['start']] * fs) : (int)(frameTime[voiceseg[m]['end']] *fs) ]
                        #计算mfcc矩阵
                        tempmfcc = librosa.feature.mfcc(y=data_seg_1, sr=fs)
                        #将特征压入数据集中去
                        r_,c_=tempmfcc.shape
                        if c_>=15 and c_<=20:
                            #vggish只接受.wav格式的数据输入，首先将我们获取到的语音段作为.wav数据临时保存到指定文件夹中去
                            #create_wav_file("C:/Users/gnyy/Desktop/Underwater_Superlimb-master/wav_data/wav/wav.wav",data_seg_1,fs)
                            #a = AudioFileClip(dir_list[n])  #读入文件
                            #audio1 = a.subclip(frameTime[voiceseg[m]['start']] ,frameTime[voiceseg[m]['end']] )   #剪切
                            #audiocct = concatenate_audioclips([audio1])
                            #audiocct .write_audiofile('C:/Users/gnyy/Desktop/Underwater_Superlimb-master/wav_data/wav/wav.wav') #输出
                            try:
                                #tempm=model.forward('C:/Users/gnyy/Desktop/Underwater_Superlimb-master/wav_data/wav/wav.wav')
                                #print(tempm)
                                if(c_>10):
                                    if c_<20:
                                        t=np.zeros((20,20-c_))
                                        tempmfcc=np.column_stack((tempmfcc,t))
                                    else:
                                        tempmfcc=tempmfcc[:,:20]
                                tempmfcc=np.transpose(tempmfcc)
                                #获取转置的mfcc矩阵，每行为一个时序的20个频段的数据分布
                                temp=[]
                                temp.append(tempmfcc)
                                #tempm=tempm.cpu()
                                #temp.append(tempm)
                                temp.append(index)
                                whole_data.append(temp)
                            except(RuntimeError):
                                print("语音段无效")
                            
        index=index+1
    return whole_data

#获取语音时序数据
def get_wav_time_data(path):
    AU = Voice_Base(path=path)
    data_two, fs, n_bits= AU.audioread()
    data= data_two[:,1]
    #去噪
    data=noise_reduce(voice_data=data)
    #数据归一化
    data=data-np.mean(data)
    data /= np.max(data)

    return data,fs,n_bits,AU

#定义自己的数据读取方式
class MyDataset(Dataset):
    def __init__(self,i,j,transform=None):
        super().__init__()
        self.wav_data=get_wav_data(i,j)
        self.i=i
        self.j=j
        self.transform=transform
    def __getitem__(self, index):
        data=self.wav_data[index][0]
        label=self.wav_data[index][1]
        label=np.array(label)
        label=torch.from_numpy(label)
        if self.transform is not None:
            data=self.transform(data)
        return data,label
    def __len__(self):
        return len(self.wav_data)

##定义RNN模型
class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=nn.LSTM(
            #语音段的时序数
            input_size=INPUT_SIZE,
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
##网络方法(RNN)

model=torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()


transform=transforms.ToTensor()
traindatasets=MyDataset(1,5,transform)
data_loader=DataLoader(traindatasets,batch_size=1,shuffle=True,num_workers=0)
#训练整批数据多少次
EPOCH=8
BATCH_SIZE=64
#时间长度
TIME_STEP=20
#TIME_STEP=16
#滤波器位数
INPUT_SIZE=20
#INPUT_SIZE=8
#学习率
LR=0.0005

#准备测试数据
test_data=MyDataset(5,5,transform)
test_lodader=DataLoader(test_data,batch_size=1,shuffle=True,num_workers=0)
rnn=RNN()

#RNN 训练
optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()
#训练
for epoch in range (EPOCH):
    for step,(x,b_y) in  enumerate(data_loader):
        b_x=x.view(-1,20,20)
        b_x=b_x.to(torch.float32)
        output=rnn(b_x)
        b_y=b_y.to(torch.long)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if step%10==0:
           print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, EPOCH, step+1, traindatasets.__len__(), loss.item()))
#测试模型
correct=0
total=0
for data,labels in test_lodader:
    data=data.view(-1,20,20)
    data=data.to(torch.float32)
    labels=labels.to(torch.long)
    output=rnn(data)
    _,predicted=torch.max(output.data,1)
    total+=labels.size(0)
    correct+=(predicted==labels).sum()
print('Test Accuracy of the model: %d %%' % (100 * correct / total))

torch.save(rnn, 'rnn.pth')

#计算时序数据
##############################
# Test the Class Methods
#获取待比较的语音段
#AU = Voice_Base(path='C:\\Users\\gnyy\\Desktop\\Underwater_Superlimb-master\\python\\script\\single_pitch\\so_4.wav')
# template="C:\\Users\\gnyy\\Desktop\\Underwater_Superlimb-master\\template\\5.txt"
# # NU=Voice_Base(path='C:\\Users\\gnyy\\Desktop\\Underwater_Superlimb-master\\wav_voice_1213\\wav_single_pitch_1215\\test_noise_sample.wav')
# #模板文件的路径，以txt形式记录，txt第一行为矩阵的行数，txt第二行为矩阵的列数，后续每一行为一个数字数据。（注：存储的矩阵为原始mcff矩阵的转置矩阵）
# File_Path=[];
# File_Path.append('C:\\Users\\gnyy\\Desktop\\Underwater_Superlimb-master\\template\\1.txt')
# File_Path.append('C:\\Users\\gnyy\\Desktop\\Underwater_Superlimb-master\\template\\2.txt')
# File_Path.append('C:\\Users\\gnyy\\Desktop\\Underwater_Superlimb-master\\template\\3.txt')
# File_Path.append('C:\\Users\\gnyy\\Desktop\\Underwater_Superlimb-master\\template\\4.txt')
# File_Path.append('C:\\Users\\gnyy\\Desktop\\Underwater_Superlimb-master\\template\\5.txt')




# #获取语音的时序数据 频率 数据深度
# data_two, fs, n_bits= AU.audioread()
# # noise_two,n_fs,noise_bits=NU.audioread()
# data= data_two[:,1]
# # noise=noise_two[:,1]
# # noise=noise[int(len(noise)/2):] 
# # data, fs, n_bits = AU.audioread()
# print(len(data))



# #打印中间数据，测试用
# #print(tempdata)
# #数据重整为一维
# data=noise_reduce(voice_data=data)
# data=data-np.mean(data)
# data /= np.max(data)
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
# #vad_specEN(data, wnd, inc, NIS, thr1, thr2, fs, self):






# voiceseg, vsl, SF, NF, Enm =vad_specEN();

# fn = len(SF)
# frameTime = AU.FrameTimeC(fn, wlen, inc, fs)
# plt.subplot(2, 1, 1)
# plt.plot(time, data)
# plt.subplot(2, 1, 2)
# plt.plot(frameTime, Enm)

# #基准模板矩阵
# basemfccs=[];
# #遍历基准模板文件，获取基准模板数组
# for i in range (5):
#     file_path=File_Path[i];
#     basemfccs.append(getBaseMcff(file_path))

# #向文件中写入模板文件数据
# # file_path=template
# # #提取某个数据作为初始模板
# # data_seg_1 = data[ (int)(frameTime[voiceseg[6]['start']] * fs) : (int)(frameTime[voiceseg[6]['end']] *fs) ]
# # basemfcc = librosa.feature.mfcc(data_seg_1, fs)

# # #遍历文件下的所有声音，计算平均模板
# # for i in range(vsl):
# #     if i>=1:
# #         data_seg_1 = data[ (int)(frameTime[voiceseg[i]['start']] * fs) : (int)(frameTime[voiceseg[i]['end']] *fs) ]
# #         tempmfcc = librosa.feature.mfcc(data_seg_1, fs)
# #         d,cost_matrix,acc_cost_matrix,path=dtw(np.transpose(basemfcc),np.transpose(tempmfcc),dist=manhattan_distance);
# #         r,c=tempmfcc.shape
# #         if c>=15:
# #             r,c=basemfcc.shape
# #             basemfcc=np.transpose(basemfcc)
# #             tempmfcc=np.transpose(tempmfcc)
# #             for j in range (c):
# #                 basemfcc[j]=(basemfcc[path[0][j]]+tempmfcc[path[1][j]])/2
# #             basemfcc=np.transpose(basemfcc)

# # with open(file_path,mode='w',encoding='utf-8') as file_obj:
# #      r,c=basemfcc.shape;
# #      file_obj.write(str(r));
# #      file_obj.write('\n')
# #      file_obj.write(str(c));
# #      file_obj.write('\n')
# #      for i in range (r):
# #          for j in range (c):
# #              file_obj.write(str(basemfcc[i][j]));
# #              file_obj.write('\n')



# index=[];
# dis=[]
# for i in range(vsl):
#     plt.subplot(2, 1, 1)
#     plt.plot(frameTime[voiceseg[i]['start']], 0, '.k')
#     plt.plot(frameTime[voiceseg[i]['end']], 0, 'or')
#     plt.legend(['signal', 'start', 'end'])
#     plt.subplot(2, 1, 2)
#     plt.plot(frameTime[voiceseg[i]['start']], 0, '.k')
#     plt.plot(frameTime[voiceseg[i]['end']], 0, 'or')
#     plt.legend(['熵谱', 'start', 'end'])
#     plt.figure(figsize=(20,10))
#     data_seg_1 = data[ (int)(frameTime[voiceseg[i]['start']] * fs) : (int)(frameTime[voiceseg[i]['end']] *fs) ]
#     time_seg_1 = np.linspace(frameTime[voiceseg[i]['start']],frameTime[voiceseg[i]['end']], len(data_seg_1))
#     # plt.plot(time_seg_1, data_seg_1)
#     mfccs = librosa.feature.mfcc(data_seg_1, fs)
#     #标定最短距离
#     mind=99999;
#     #标定类别
#     minindex=-1;
#     #遍历所有模板
#     for i in range (5):
#         d,cost_matrix,acc_cost_matrix,path=dtw(np.transpose(mfccs),basemfccs[i],dist=manhattan_distance);
#         print("当前信号与模板类别{}的差值为:".format(i+1),"{}".format(d));
#         if d<mind:
#             mind=d;
#             minindex=i;
#     ##print("当前信号与模板类别{}差值最小".format(minindex+1),"差值为{}".format(mind));
#     index.append(minindex+1)
#     dis.append(mind)
#     plt.close()
# print(index)
# print(dis)
# plt.show()





# plt.savefig('images/TwoThr.png')
####

##############################    

