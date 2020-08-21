
#include<fstream>
#include<thread>
#include<random>
#include<time.h>
#include<Windows.h>
#include<iostream>
#include<cstdio>
#include <immintrin.h>
#include <algorithm>
#include "Header.h"

using namespace std;

struct neuron {
	double value;//значение
	double error;//ошибка
	void act() {
		value = (1 / (1 + pow(2.71828, -value)));//сигмоидная функция активации
	}
};

struct data_one//картинки
{
	double info[4096];
	char rresult;
};

class network {
public:
	int layers;//кол во слоев
	neuron** neurons;//нейроны
	double*** weights;// 1 слой   2 носер нейрона  3 носер связи
	int* size;//динамический масив кол во нейронов в слое 
	int threadsNum = 1;//кол во потоков
	double sigm_pro(double x)//производная от сигмоидной функции 
	{
		if ((fabs(x - 1) < 1e-9) || (fabs(x) < 1e-9)) return 0.0;
		double res = x * (1.0 - x);
		return res;
	}
	double predict(double x)//принимает значение нейрона из выходного слоя
	{
		if (x >= 0.8) {
			return 1;
		}
		else {
			return 0;
		}
	}
	void setLayersNotStudy(int n, int* p, string filename)//слои  кол во нейронов имя файла 
	{
		ifstream fin;
		fin.open(filename);
		srand(time(0));
		layers = n;
		neurons = new neuron * [n];
		weights = new double** [n - 1];
		size = new int[n];
		for (int i = 0; i < n; i++) {
			size[i] = p[i];
			neurons[i] = new neuron[p[i]];
			if (i < n - 1) {
				weights[i] = new double* [p[i]];
				for (int j = 0; j < p[i]; j++) {
					weights[i][j] = new double[p[i + 1]];
					for (int k = 0; k < p[i + 1]; k++) {
						fin >> weights[i][j][k];
					}
				}
			}
		}
	}
	void setLayers(int n, int* p)//принимает кол во слоев и нейронов ,генерирует рандомные веса
	{
		srand(time(0));
		layers = n;
		neurons = new neuron * [n];
		weights = new double** [n - 1];
		size = new int[n];
		for (int i = 0; i < n; i++) {
			size[i] = p[i];
			neurons[i] = new neuron[p[i]];
			if (i < n - 1) {
				weights[i] = new double* [p[i]];
				for (int j = 0; j < p[i]; j++) {
					weights[i][j] = new double[p[i + 1]];
					for (int k = 0; k < p[i + 1]; k++) {
						weights[i][j][k] = ((rand() % 100)) * 0.01 / size[i];
					}
				}
			}
		}
	}
	
	void set_input(double p[]) // подает входные данные для нейронной сети
	{
		for (int i = 0; i < size[0]; i++) {
			neurons[0][i].value = p[i];
		}
	}
	

	void LayersCleaner(int LayerNumber, int start, int stop)//вспомогательная функция BP для очистки слоев 
 {
		srand(time(0));
		for (int i = start; i < stop; i++) {
			neurons[LayerNumber][i].value = 0;
		
		}
	}

	void ForwardFeeder(int LayerNumber, int start, int stop)//проходит по каждому нейрону и считает его значение 
	{
		for (int j = start; j < stop; j++) {
			for (int k = 0; k < size[LayerNumber - 1]; k++) {
				neurons[LayerNumber][j].value += neurons[LayerNumber - 1][k].value * weights[LayerNumber - 1][k][j];
			}
		
			neurons[LayerNumber][j].act();
		}
	}

	
	double ForwardFeed() //вызывает  LayersCleaner и ForwardFeeder
	{
		setlocale(LC_ALL, "ru");
		//cout << "Function ForwardFeed:\n";
		//cout << "Threads: " << threadsNum << endl;
		for (int i = 1; i < layers; i++) {
			if (threadsNum == 1) {
				
				thread th1([&]() {
					LayersCleaner(i, 0, size[i]);
					});
				th1.join();
			}
			
			if (threadsNum == 1) 
			{
				thread th1([&]() 
					{
					ForwardFeeder(i, 0, size[i]);
					});
				th1.join();
			}
			
			
		}
		double max = 0;
		double prediction = 0;
		for (int i = 0; i < size[layers - 1]; i++) //нахождение максимального значения последнего слоя 
		{
			cout << char(i + 65) << " : " << neurons[layers - 1][i].value << endl;
			if (neurons[layers - 1][i].value > max)
			{
				max = neurons[layers - 1][i].value;
				prediction = i;
			}
		}
		return prediction;
	}
	
	void ErrorCounter(int LayerNumber, int start, int stop, double prediction, double rresult, double lr) //считает ошибку для каждого нейрона
	{
		if (LayerNumber == layers - 1) //если номер  правильного  результата  не равен номеру нейрона 
		{
			for (int j = start; j < stop; j++) {
				if (j != int(rresult)) {
					neurons[LayerNumber][j].error = -pow((neurons[LayerNumber][j].value),2);
				}
				else {
					neurons[LayerNumber][j].error = pow(1.0 - neurons[LayerNumber][j].value, 2);
				}
			}
		}
		else {
			for (int j = start; j < stop; j++) {
				double error = 0.0;
				for (int k = 0; k < size[LayerNumber + 1]; k++) {
					error += neurons[LayerNumber + 1][k].error * weights[LayerNumber][j][k];
				}
				neurons[LayerNumber][j].error = error;
			}
		}

	}

	void WeightsUpdater(int start, int stop, int LayerNum, int lr) //обновляет веса после BP
	{
		int i = LayerNum;
		for (int j = start; j < stop; j++) {
			for (int k = 0; k < size[i + 1]; k++) {
				weights[i][j][k] += lr * neurons[i + 1][k].error * sigm_pro(neurons[i + 1][k].value) * neurons[i][j].value;
			}
		}
	}

	void BackPropogation(double prediction, double rresult, double lr)//результат нейронной сети правильный результат
	{
		for (int i = layers - 1; i > 0; i--) {
			if (threadsNum == 1) {
				if (i == layers - 1) {
					for (int j = 0; j < size[i]; j++) {
						if (j != int(rresult)) {
							neurons[i][j].error = -pow((neurons[i][j].value), 2);
						}
						else {
							neurons[i][j].error = pow(1.0 - neurons[i][j].value, 2);
						}

					}
				}
				else 
				{
					for (int j = 0; j < size[i]; j++)
					{
						double error = 0.0;
						for (int k = 0; k < size[i + 1]; k++)
						{
							error += neurons[i + 1][k].error * weights[i][j][k];
						}
						neurons[i][j].error = error;
					}
				}
			}
			
			
		}
		for (int i = 0; i < layers - 1; i++)
		{
			if (threadsNum == 1) 
			{
				for (int j = 0; j < size[i]; j++)
				{
					for (int k = 0; k < size[i + 1]; k++)
					{
						weights[i][j][k] += lr * neurons[i + 1][k].error * sigm_pro(neurons[i + 1][k].value) * neurons[i][j].value;
					}
				}
			}
			
			
		}
	}

	bool SaveWeights()//открывает файл и сохранят в него веса
	{
		ofstream fout;
		fout.open("weights.txt");
		for (int i = 0; i < layers; i++) {
			if (i < layers - 1) {
				for (int j = 0; j < size[i]; j++) {
					for (int k = 0; k < size[i + 1]; k++) {
						fout << weights[i][j][k] << " ";
					}
				}
			}
		}
		fout.close();
		return 1;
	}
};




int main() {

	srand(time(0));
	setlocale(LC_ALL, "Russian");
	ifstream fin;//ввод
	ofstream fout;//вывод
	fout.open("log.txt");
	const int l = 4;//слоев в нейросети 
	const int input_l = 4096;//64 пикселей 
	int size[l] = { input_l, 256, 64,   26 };
	network nn;//нейросеть

	double input[input_l];//входные значения нейрочети 

	char rresult;//правильный ответ
	double result;//номер нейрона с максимальным значением
	double ra = 0;//правильных ответов за эпоху
	int maxra = 0;
	int maxraepoch = 0;
	const int n = 83;//обучение нейросети
	
	

	
	bool to_study = 0;
	//std::cout << "Производить обучение?";
	//std::cin >> to_study;
	

	data_one* data = new data_one[n];

	if (to_study) {
		fin.open("lib.txt");//вводятся значения из файла 
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < input_l; j++) {
				fin >> data[i].info[j];
			}
			fin >> data[i].rresult;//вводится правильный ответ
			data[i].rresult -= 65;
		}

		nn.setLayers(l, size);//создаем слои 
		for (int e = 0; ra / n * 100 < 100; e++)//цикл до тех пор пока сеть не угадает 100% букв 
		{

			fout << "Epoch # " << e << endl;//какая эпоха
			
			ra = 0;
			double w_delta = 0;



			for (int i = 0; i < n; i++) {

				for (int j = 0; j < 4096; j++) {
					input[j] = data[i].info[j];
				}
				rresult = data[i].rresult;

				nn.set_input(input);
				

				result = nn.ForwardFeed();

				

				if (result == rresult)//если переменная равна правильному ответу 
				{

					cout << "Угадал букву " << char(rresult + 65) << "\t\t\t****" << endl;
					ra++;
				}
				else//если не угадал
				{

					
					nn.BackPropogation(result, rresult, 0.5);
					

				}
			}

			
			cout << "Right answers: " << ra / n * 100 << "% \t Max RA: " << double(maxra) / n * 100 << "(epoch " << maxraepoch << " )" << endl;
			

			if (ra > maxra) {
				maxra = ra;
				maxraepoch = e;
			}
			if (maxraepoch < e - 250) {
				maxra = 0;
			}
		}

		if (nn.SaveWeights()) {
			cout << "Веса сохранены!";
		}
	}
	else {
		nn.setLayersNotStudy(l, size, "weights.txt");
	}
	
	nn.setLayersNotStudy(l, size, "weights.txt");
	
	fin.close();
	
	
	bool to_start_test ;
	cout << " Угадать букву  ? :  ";
	cin >> to_start_test;
	char right_res;
	while (to_start_test)
	{
		
		if (to_start_test) {
			letter();
			fin.open("letter.txt");

			for (int i = 0; i < input_l; i++) {
				fin >> input[i];
			}
			nn.set_input(input);
			result = nn.ForwardFeed();
			cout << "Я считаю, что это буква " << char(result + 65) << "\n\n";
			cout << "А какая это буква на самом деле?...";
			cin >> right_res;
			if (right_res != result + 65) {
				cout << "Хорошо , исправляю ошибку!\n";
				nn.BackPropogation(result, right_res - 65, 0.15);
				nn.SaveWeights();
			}
		}
		fin.close();
		cout << " Угадать букву  ? :  ";
		cin >> to_start_test;
	}
	

	

	
	return 0;
}
