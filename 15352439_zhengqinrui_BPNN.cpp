#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <time.h>
#include <fstream>
#include <sstream>
using namespace std;

#define innode 22
#define hidenode1 22
#define hidenode2 22
#define hidelayer 2 
#define outnode 1 
#define rows 8619

double learningrate[3]={0.0001,0.002,0.003};



inline double get_11Random()    // -1 ~ 1
{
    return ((2.0*(double)rand()/RAND_MAX) - 1);
}

double f1(double x)
{
	return (exp(x)-exp(-x))/(exp(x)+exp(-x));
}

double f2(double x)
{
	if(x>=0)return x;
	else return 0.1*x;
}

double f3(double x)
{
	if(x>=0)return x;
	else return 0.01*x;
}

double df1(double x)
{
	return 1-f1(x)*f1(x);
}

double df2(double x)
{
	if(x>=0)return 1;
	else return 0.1;
}

double df3(double x)
{
	if(x>=0)return 1;
	else return 0.01;
}
// --- �����ڵ㡣�������·�����--- 
// 1.value:     �̶�����ֵ�� 
// 2.weight:    ��Ե�һ��������ÿ���ڵ㶼��Ȩֵ�� 
// 3.wDeltaSum: ��Ե�һ��������ÿ���ڵ�Ȩֵ��deltaֵ�ۻ�
typedef struct inputNode
{
    double value;
    vector<double> weight, wDeltaSum;
}inputNode;

// --- �����ڵ㡣����������ֵ��--- 
// 1.value:     �ڵ㵱ǰֵ�� 
// 2.delta:     ����ȷ���ֵ֮���deltaֵ�� 
// 3.rightout:  ��ȷ���ֵ
// 4.bias:      ƫ����
// 5.bDeltaSum: bias��deltaֵ���ۻ���ÿ���ڵ�һ��
typedef struct outputNode   // �����ڵ�
{
    double value, delta, rightout, bias, bDeltaSum,z;
}outputNode;

// --- ������ڵ㡣����������ֵ��--- 
// 1.value:     �ڵ㵱ǰֵ�� 
// 2.delta:     BP�Ƶ�����deltaֵ��
// 3.bias:      ƫ����
// 4.bDeltaSum: bias��deltaֵ���ۻ���ÿ���ڵ�һ��
// 5.weight:    �����һ�㣨������/����㣩ÿ���ڵ㶼��Ȩֵ�� 
// 6.wDeltaSum�� weight��deltaֵ���ۻ��������һ�㣨������/����㣩ÿ���ڵ���Ի���
typedef struct hiddenNode   // ������ڵ�
{
    double value, delta, bias, bDeltaSum,z;
    vector<double> weight, wDeltaSum;
}hiddenNode;

// --- �������� --- 
typedef struct sample
{
    vector<double> in, out;
}sample;

class BPNN
{
public:
    BPNN();    //���캯��
    void forwardPropagationEpoc();  // ��������ǰ�򴫲�
    void backPropagationEpoc();     // �����������򴫲�

    void training (vector<sample> sampleGroup, double times);// ���� weight, bias
    void predict  (vector<sample>& testGroup);                          // ������Ԥ��

    void setInput (vector<double> sampleIn);     // ����ѧϰ��������
    void setOutput(vector<double> sampleOut);    // ����ѧϰ�������

public:
	double error;
    inputNode* inputLayer[innode];           // �����
    outputNode* outputLayer[outnode];        // �����
    hiddenNode* hiddenLayer1[hidenode1]; 	 // ������
	hiddenNode* hiddenLayer2[hidenode2];      // ������
};

BPNN::BPNN()
{
	srand((unsigned)time(NULL));        // ���������    
	error=20000;
    // ��ʼ�������
    for(int i=0;i<innode;i++)
    {
        inputLayer[i]=new inputNode();
        for(int j=0;j<hidenode1;j++) 
        {
            inputLayer[i]->weight.push_back(get_11Random());
            inputLayer[i]->wDeltaSum.push_back(0);
        }
    }

    
    for(int i=0;i<hidenode1;i++)
    {
        hiddenLayer1[i]=new hiddenNode();
        hiddenLayer1[i]->bias=get_11Random();
        for(int j=0;j<hidenode2;j++) 
        {
            hiddenLayer1[i]->weight.push_back(get_11Random());
            hiddenLayer1[i]->wDeltaSum.push_back(0);
        }
    }
	for(int i=0;i<hidenode2;i++)
	{
		hiddenLayer2[i]=new hiddenNode();
		hiddenLayer2[i]->bias=get_11Random();
		for(int j=0;j<outnode;j++)
		{
			hiddenLayer2[i]->weight.push_back(get_11Random());
			hiddenLayer2[i]->wDeltaSum.push_back(0);
		}
	}
    // ��ʼ�������
    for(int i=0;i<outnode;i++)
    {
        outputLayer[i]=new outputNode();
        outputLayer[i]->bias=get_11Random();
    }
}


void BPNN::forwardPropagationEpoc()
{
    for(int i=0;i<hidenode1;i++)
    {
        double sum=0;
        for(int j=0;j<innode;j++) 
        {
            sum+=inputLayer[j]->value*inputLayer[j]->weight[i];
        }
        sum+=hiddenLayer1[i]->bias;
        hiddenLayer1[i]->z=sum;
        hiddenLayer1[i]->value=f1(sum);
    }
       
    for(int i=0;i<hidenode2;i++)
    {
        double sum=0;
        for(int j=0;j<hidenode1;j++) 
        {
            sum+=hiddenLayer1[j]->value*hiddenLayer1[j]->weight[i];
        }
        sum+=hiddenLayer2[i]->bias;
        hiddenLayer2[i]->z=sum; 
        hiddenLayer2[i]->value=f2(sum);
    }

    for(int i=0;i<outnode;i++)
    {
        double sum=0;
        for(int j=0;j<hidenode2;j++)
        {
            sum+=hiddenLayer2[j]->value*hiddenLayer2[j]->weight[i];
        }
        sum+=outputLayer[i]->bias;
        outputLayer[i]->z=sum;
        outputLayer[i]->value=f3(sum);
    }
   
}

void BPNN::backPropagationEpoc()
{
    for(int i=0;i<outnode;i++)
    {
        double tmpe=fabs(outputLayer[i]->value-outputLayer[i]->rightout);
        error+=tmpe*tmpe/(2*rows);
        outputLayer[i]->delta=(outputLayer[i]->value-outputLayer[i]->rightout)*df3(outputLayer[i]->z);
    }
    
    for(int i=0;i<hidenode2;i++)
    {
        double sum=0;
        for(int j=0;j<outnode;j++)
		{
			sum+=outputLayer[j]->delta*hiddenLayer2[i]->weight[j];
		}
        hiddenLayer2[i]->delta=sum*df2(hiddenLayer2[i]->z);
    }
            
	for(int i=0;i<hidenode1;i++)
    {
        double sum=0;
        for(int j=0;j<hidenode2;j++)
		{
			sum+=hiddenLayer2[j]->delta*hiddenLayer1[i]->weight[j];
		}
        hiddenLayer1[i]->delta=sum*df1(hiddenLayer1[i]->z);
    }

    for(int i=0;i<innode;i++)
    {
        for(int j=0;j<hidenode1;j++)
        {
            inputLayer[i]->wDeltaSum[j]+=inputLayer[i]->value*hiddenLayer1[j]->delta;
        }
    }

    for(int i=0;i<hidenode2;i++)
    {
        hiddenLayer2[i]->bDeltaSum+=hiddenLayer2[i]->delta;
        for(int j=0;j<outnode;j++)
        {
			 hiddenLayer2[i]->wDeltaSum[j]+=hiddenLayer2[i]->value*outputLayer[j]->delta;
		}
    }
    for(int i=0;i<hidenode1;i++)
    {
        hiddenLayer1[i]->bDeltaSum+=hiddenLayer1[i]->delta;
        for (int j=0;j<hidenode2;j++)
        { 
			hiddenLayer1[i]->wDeltaSum[j]+=hiddenLayer1[i]->value*hiddenLayer2[j]->delta;
		}
    }
       
    for(int i=0;i<outnode;i++)outputLayer[i]->bDeltaSum+=outputLayer[i]->delta;
}

void BPNN::training(vector<sample> sampleGroup, double times)
{
	int sampleNum = sampleGroup.size();

	ofstream fout("C:\\Users\\BurNInglove\\Desktop\\BPNN.txt");
	if(!fout)
	{
		cout<<"error"<<endl;
		return ; 
	}
    while(times--)
    {
    	
        fout << error << endl;
        error = 0;
        
        for(int i=0;i<innode;i++)inputLayer[i]->wDeltaSum.assign(inputLayer[i]->wDeltaSum.size(),0);
        for(int i=0;i<hidenode1;i++)hiddenLayer1[i]->wDeltaSum.assign(hiddenLayer1[i]->wDeltaSum.size(),0);
        for(int i=0;i<hidenode2;i++)hiddenLayer2[i]->wDeltaSum.assign(hiddenLayer2[i]->wDeltaSum.size(),0);
        for(int i=0;i<outnode;i++)outputLayer[i]->bDeltaSum=0;
		
	
        for(int i=0;i<sampleNum;i++)
        {
            setInput(sampleGroup[i].in);
            setOutput(sampleGroup[i].out);
			
            forwardPropagationEpoc();
           
            backPropagationEpoc();
        }

        for(int i=0;i<innode;i++)
        {
            for(int j=0;j<hidenode1;j++) 
            {
                inputLayer[i]->weight[j]-=learningrate[0]*(inputLayer[i]->wDeltaSum[j]/sampleNum+0.01*inputLayer[i]->weight[j]);
            }
        }

		for(int i=0;i<hidenode1;i++)
        {
            hiddenLayer1[i]->bias-=learningrate[0]*hiddenLayer1[i]->bDeltaSum/sampleNum;
            for(int j=0;j<hidenode2;j++) 
            { 
				hiddenLayer1[i]->weight[j]-=learningrate[1]*(hiddenLayer1[i]->wDeltaSum[j]/sampleNum+0.01*hiddenLayer1[i]->weight[j]); 
			}
        }
        for(int i=0;i<hidenode2;i++)
        { 
            hiddenLayer2[i]->bias-=learningrate[1]*hiddenLayer2[i]->bDeltaSum/sampleNum;
            for(int j=0;j<outnode;j++) 
            {
			 	hiddenLayer2[i]->weight[j]-=learningrate[2]*(hiddenLayer2[i]->wDeltaSum[j]/sampleNum+0.01*hiddenLayer2[i]->weight[j]);
			}
        }
        
        for(int i=0;i<outnode;i++)
        { 
			outputLayer[i]->bias-=learningrate[2]*outputLayer[i]->bDeltaSum/sampleNum;
		}
    }
    fout.close();
}

void BPNN::predict(vector<sample>& testGroup)
{
    int size=testGroup.size();
    for(int t=0;t<size;t++)
    {
        testGroup[t].out.clear();
        setInput(testGroup[t].in);
        for(int j=0;j<hidenode1;j++)
        {
            double sum=0;
            for(int k=0;k<innode;k++) 
            {
                sum+=inputLayer[k]->value*inputLayer[k]->weight[j];
            }
            sum+=hiddenLayer1[j]->bias;
            hiddenLayer1[j]->value=f1(sum);
        }
        for(int j=0;j<hidenode2;j++)
        {
            double sum=0;
            for (int k=0;k<hidenode1;k++) 
            {
                sum+=hiddenLayer1[k]->value*hiddenLayer1[k]->weight[j];
            }
            sum+=hiddenLayer2[j]->bias;
            hiddenLayer2[j]->value=f2(sum);
        }

        for (int i=0;i<outnode;i++)
        {
            double sum=0;
            for (int j=0;j<hidenode2;j++)
            {
                sum+=hiddenLayer2[j]->value*hiddenLayer2[j]->weight[i];
            }
            sum+=outputLayer[i]->bias;
            outputLayer[i]->value=f3(sum);
            testGroup[t].out.push_back(outputLayer[i]->value);
        }
    }
}

void BPNN::setInput(vector<double> sampleIn)
{
    for(int i=0;i<innode;i++)inputLayer[i]->value=sampleIn[i];
}

void BPNN::setOutput(vector<double> sampleOut)
{
    for(int i=0;i<outnode;i++)outputLayer[i]->rightout=sampleOut[i];
}


int main()
{
	BPNN testNet;
	sample sampleInOut[8625];
	sample testInOut[510];
    // ѧϰ����
    vector<double> samplein[8625];
    vector<double> sampleout[8625];
    vector<double> testin[510];
    vector<double> testout[510];
    fstream fin;
	fin.open("C:\\Users\\BurNInglove\\Documents\\Tencent Files\\779401896\\FileRecv\\train.csv",ios::in);
	if(!fin)
	{
		cout<<"File open error!\n";
		return 0;
	}
	string read;
	long long judge=0;
	int cnt_r=0;

	while(getline(fin, read, '\n'))
	{
		stringstream stream(read);
		string split;
		while(getline(stream, split, ','))
		{
			if(judge%23==22)
			{
				const char *a=split.c_str();
				double d=atof(a);
				sampleout[cnt_r].push_back(d);
			}
			else
			{
				const char *a=split.c_str();
				double d=atof(a);
				samplein[cnt_r].push_back(d);
			}	
			judge++;
		}
    	cnt_r++;
    }
    fin.close();
    cout<<"over"<<endl;
    for (int i=0;i<8619;i++)
    {
        sampleInOut[i].in=samplein[i];
       
        sampleInOut[i].out=sampleout[i];

    }
    vector<sample> sampleGroup(sampleInOut, sampleInOut+8619);
    testNet.training(sampleGroup,5000);
	cout<<"end"<<endl;
    
    fin.close();
    fstream fin_;
    fin_.open("C:\\Users\\BurNInglove\\Documents\\Tencent Files\\779401896\\FileRecv\\test.csv",ios::in);
    string read_;
	long long judge_=0;
	cnt_r=0;

	while(getline(fin_, read_, '\n'))
	{
		stringstream stream(read_);
		string split;
		while(getline(stream, split, ','))
		{
			if(judge_%23==22)
			{
				
			}
			else
			{
				const char *a=split.c_str();
				double d=atof(a);
				testin[cnt_r].push_back(d);
			}	
			judge_++;
		}
    	cnt_r++;
    }
    for (int i = 0; i < 504; i++) testInOut[i].in = testin[i];
    vector<sample> testGroup(testInOut, testInOut + 504);

    // Ԥ��������ݣ���������
    ofstream fout("C:\\Users\\BurNInglove\\Desktop\\testk.txt");
	if(!fout)
	{
		cout<<"error"<<endl;
		return 0; 
	}
    testNet.predict(testGroup);
    for (int i = 0; i < testGroup.size(); i++)
    {
         for (int j = 0; j < testGroup[i].out.size(); j++) fout << testGroup[i].out[j] <<endl;
    }
		
    system("pause");
    return 0;
}
