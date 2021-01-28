using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace AI_Laba_NeuralNet
{
    class Program
    {
        public static Random r = new Random((int)DateTime.Now.Ticks);
        static void Main(string[] args)
        {
            string filename = "iris.csv";

            double[][] TestX;
            int[] TestY;

            var data = DataGet(filename);
            data = SplitData(data, out TestX, out TestY, 50);
            var dataTrain = DataFit(data);

            var NN = new NeuralNetwork();
            NN.Layers.Add(new Layer(3));
            NN.Layers.Add(new Layer(3));
            NN.Build(data.X[0].Length, Sigmoid, dSigmooid);
            //NN.printWeights();
            DateTime start = DateTime.Now;
            NN.Train(dataTrain.X, dataTrain.Y, learningRate: 0.02, MaxIteration: 1000);
            var stop = DateTime.Now - start;

            Console.WriteLine($"Training complete, time elapsed - {stop.TotalMilliseconds} miliseconds");

            NN.Test(TestX, TestY, verbose: true);

            //NN.printWeights();

            Console.ReadKey();
        }
        static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
        static double dSigmooid(double x) => x * (1 - x);
        static public (double[][] X, double[] Y) DataGet(string filePath, char sep = ';')
        {
            var listX = new List<double[]>();
            var listY = new List<double>();

            string[] data = File.ReadAllLines(filePath);
            for (int i = 1; i < data.Length; i++)
            {
                var subData = data[i].Split(sep);
                double[] xrow = new double[subData.Length - 1];
                for (int j = 0; j < xrow.Length; j++)
                {
                    xrow[j] = double.Parse(subData[j]);
                }
                listX.Add(xrow);
                listY.Add(int.Parse(subData.Last()));
            }
            return (listX.ToArray(), listY.ToArray());
        }
        static public (double[][] X, double[][]Y) DataFit((double[][] X, double[] Y) data)
        {
            (double[][] X, double[][] Y) result;
            result.X = data.X;
            double[][] Y = new double[data.Y.Length][];
            for(int i = 0; i < Y.Length; i++)
            {
                Y[i] = new double[(int)data.Y.Max() + 1];
                Y[i][(int)data.Y[i]] = 1;
            }
            result.Y = Y;
            return result;
        }
        static public (double[][] X, double[] Y) SplitData((double[][] X, double[] Y) Data, out double[][] TestX, out int[] TestY, int TrainDataSplit = 80)
        {
            if (TrainDataSplit > 100 || TrainDataSplit < 0)
                throw new ArgumentOutOfRangeException();

            var TrainSize = (int)Map(0, 100, 0, Data.X.Length, TrainDataSplit);
            double[][] TrainX = new double[TrainSize][];

            for (int i = 0; i < TrainSize; i++)
                TrainX[i] = new double[Data.X[i].Length];

            double[] TrainY = new double[TrainSize];
            List<int> randList = new List<int>();

            do
            {
                var val = r.Next(0, Data.X.Length);
                if (!randList.Contains(val))
                    randList.Add(val);
            }
            while (randList.Count < TrainSize);

            for (int i = 0; i < TrainSize; i++)
            {
                for (int j = 0; j < Data.X[i].Length; j++)
                {
                    TrainX[i][j] = Data.X[randList[i]][j];
                }
                TrainY[i] = (int)Data.Y[randList[i]];
            }

            TestX = new double[Data.X.Length - TrainX.Length][];
            TestY = new int[Data.Y.Length - TrainY.Length];
            List<int> randList2 = new List<int>();

            do
            {
                var val = r.Next(0, Data.X.Length);
                if (!randList.Contains(val))
                    randList2.Add(val);
            }
            while (randList2.Count < Data.X.Length - TrainX.Length);

            for (int i = 0; i < randList2.Count; i++)
            {
                TestX[i] = Data.X[randList2[i]];
                TestY[i] = (int)Data.Y[randList2[i]];
            }

            return (TrainX, TrainY);
        }
        static double Map(double a0, double a1, double b0, double b1, double a) => b0 + (b1 - b0) * ((a - a0) / (a1 - a0));
    }
}
