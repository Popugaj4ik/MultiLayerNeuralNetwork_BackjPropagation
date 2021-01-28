using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AI_Laba_NeuralNet
{
    class NeuralNetwork
    {
        public List<Layer> Layers { get; private set; }
        public bool isBuild { get; private set; }
        public NeuralNetwork()
        {
            Layers = new List<Layer>();
        }
        public void Build(int inputSize, Func<double, double> activationFunctions, Func<double, double> derivitiveFunction)
        {
            if (Layers.Count == 0)
                throw new Exception($"No layers in neural nwtwork");
            if (isBuild)
                return;
            Layers[0].build(inputSize, activationFunctions, derivitiveFunction);
            if (Layers.Count > 1)
            {
                for (int i = 1; i < Layers.Count; i++)
                {
                    Layers[i].build(Layers[i - 1].cells.Length, activationFunctions, derivitiveFunction);
                }
            }
            isBuild = true;
        }
        public void Build(int inputSize, Func<double, double>[] activationFunctions, Func<double, double>[] derivitiveFunction)
        {
            if (activationFunctions.Length != derivitiveFunction.Length && activationFunctions.Length != Layers.Count)
                throw new Exception($"No much functions");
            if (Layers.Count == 0)
                throw new Exception($"No layers in neural nwtwork");
            if (isBuild)
                return;
            Layers[0].build(inputSize, activationFunctions[0], derivitiveFunction[0]);
            if (Layers.Count > 1)
            {
                for (int i = 1; i < Layers.Count; i++)
                {
                    Layers[i].build(Layers[i - 1].cells.Length, activationFunctions[i], derivitiveFunction[i]);
                }
            }
            isBuild = true;
        }
        public void Build(int inputSize, Func<double, double>[][] activationFunctions, Func<double, double>[][] derivitiveFunction)
        {
            if (activationFunctions.Length != derivitiveFunction.Length && activationFunctions.Length != Layers.Count)
                throw new Exception($"No much functions");
            if (Layers.Count == 0)
                throw new Exception($"No layers in neural nwtwork");
            if (isBuild)
                return;
            Layers[0].build(inputSize, activationFunctions[0], derivitiveFunction[0]);
            if (Layers.Count > 1)
            {
                for (int i = 1; i < Layers.Count; i++)
                {
                    Layers[i].build(Layers[i - 1].cells.Length, activationFunctions[i], derivitiveFunction[i]);
                }
            }
            isBuild = true;
        }
        public void Build(int inputSize, Func<double, double> activationFunctions, Func<double, double> derivitiveFunction, double bias)
        {
            if (Layers.Count == 0)
                throw new Exception($"No layers in neural nwtwork");
            if (isBuild)
                return;
            Layers[0].build(inputSize, activationFunctions, derivitiveFunction);
            if (Layers.Count > 1)
            {
                for (int i = 1; i < Layers.Count; i++)
                {
                    Layers[i].build(Layers[i - 1].cells.Length, activationFunctions, derivitiveFunction, bias);
                }
            }
            isBuild = true;
        }
        public void Build(int inputSize, Func<double, double>[] activationFunctions, Func<double, double>[] derivitiveFunction, double[] bias)
        {
            if (activationFunctions.Length != derivitiveFunction.Length && activationFunctions.Length != bias.Length && activationFunctions.Length != Layers.Count)
                throw new Exception($"No much functions");
            if (Layers.Count == 0)
                throw new Exception($"No layers in neural nwtwork");
            if (isBuild)
                return;
            Layers[0].build(inputSize, activationFunctions[0], derivitiveFunction[0]);
            if (Layers.Count > 1)
            {
                for (int i = 1; i < Layers.Count; i++)
                {
                    Layers[i].build(Layers[i - 1].cells.Length, activationFunctions[i], derivitiveFunction[i], bias[i]);
                }
            }
            isBuild = true;
        }
        public void Build(int inputSize, Func<double, double>[][] activationFunctions, Func<double, double>[][] derivitiveFunction, double[][] bias)
        {
            if (activationFunctions.Length != derivitiveFunction.Length && activationFunctions.Length != bias.Length && activationFunctions.Length != Layers.Count)
                throw new Exception($"No much functions");
            if (Layers.Count == 0)
                throw new Exception($"No layers in neural nwtwork");
            if (isBuild)
                return;
            Layers[0].build(inputSize, activationFunctions[0], derivitiveFunction[0]);
            if (Layers.Count > 1)
            {
                for (int i = 1; i < Layers.Count; i++)
                {
                    Layers[i].build(Layers[i - 1].cells.Length, activationFunctions[i], derivitiveFunction[i]);
                }
            }
            isBuild = true;
        }
        public void Train(double[][] X, double[][] Y, double learningRate = 0.5, int MaxIteration = 50)
        {
            if (!isBuild)
                throw new Exception("Network isn't built");
            if (Y[0].Length != Layers.Last().cells.Length)
                throw new Exception($"Wrong numbers of training classes and output cells\n\tNumber of training classes : {Y[0].Length}\n\tNumber of output cells : {Layers.Last().cells.Length}");
            int epoch = 0;
            var deltas = new double[Layers.Count][][];
            for (int i = 0; i < deltas.Length; i++)
            {
                deltas[i] = new double[Layers[i].cells.Length][];
                for (int j = 0; j < deltas[i].Length; j++)
                {
                    deltas[i][j] = new double[Layers[i].cells[j].weights.Length];
                }
            }
            var deltaBias = new double[Layers.Count][];
            for(int i = 0; i < deltaBias.Length; i++)
            {
                deltaBias[i] = new double[Layers[i].cells.Length];
            }
            var sums = new double[Layers.Count][];
            for (int i = 0; i < sums.Length; i++)
            {
                sums[i] = new double[Layers[i].cells.Length];
            }
            var errors = new double[Layers.Count][];
            for (int i = 0; i < sums.Length; i++)
            {
                errors[i] = new double[Layers[i].cells.Length];
            }
            var outputAll = new double[Layers.Count() + 1][];
            for(int i=1;i< outputAll.Length; i++)
            {
                outputAll[i] = new double[Layers[i - 1].cells.Length];
            }
            while (epoch < MaxIteration)
            {
                for (int i = 0; i < X.Length; i++)
                {
                    //calculatins sums and outputs
                    outputAll[0] = X[i];

                    for (int j = 0; j < Layers.Count; j++)
                    {
                        for (int k = 0; k < Layers[j].cells.Length; k++)
                        {
                            sums[j][k] = Layers[j].cells[k].weights.Select((x, w) => outputAll[j][w] * x).Sum();
                        }
                        outputAll[j + 1] = Layers[j].cells.Select((x, k) => x.activationFunction(sums[j][k] + x.bias)).ToArray();
                    }

                    var eTotal = outputAll.Last().Select((x, j) => (Math.Pow(Y[i][j] - x, 2) * (1.0 / Layers.Last().cells.Length))).Sum();

                    double[] lastD = new double[Layers.Last().cells.Length];
                    //calculations for output layer
                    for(int k = 0; k < Layers.Last().cells.Length; k++)
                    {
                        double dETotal_douty = -(Y[i][k] - outputAll.Last()[k]);
                        double dOuty_dy = Layers.Last().cells[k].derivitiveFunction(outputAll.Last()[k]);
                        lastD[k] = dETotal_douty * dOuty_dy;
                        for (int w = 0; w < Layers.Last().cells[k].weights.Length; w++)
                        {
                            double dy_dw = outputAll[outputAll.Length - 2][w];
                            double dEtotal_dw = lastD[k] * dy_dw;
                            deltas.Last()[k][w] = -learningRate * dEtotal_dw;
                        }
                        deltaBias.Last()[k] = -learningRate * lastD[k];
                    }

                    //calculations for hidden layers
                    for(int j = Layers.Count - 2; j >= 0; j--)
                    {
                        double[] nowD = new double[Layers[j].cells.Length];
                        for(int k = 0; k < Layers[j].cells.Length; k++)
                        {//calculation cell error
                            double dETotal_dOutH = 0;
                            for (int kk = 0; kk < Layers[j + 1].cells.Length; kk++)
                            {
                                dETotal_dOutH += lastD[kk] * Layers[j + 1].cells[kk].weights[k];
                            }
                            double dOutH_dH = Layers[j].cells[k].derivitiveFunction(outputAll[j + 1][k]);
                            nowD[k] = dETotal_dOutH * dOutH_dH;
                            for (int w = 0; w < Layers[j].cells[k].weights.Length; w++)
                            {
                                double dH_dW = outputAll[j][w];
                                double dEtotal_dW = nowD[k] * dH_dW;
                                deltas[j][k][w] = -learningRate * dEtotal_dW;
                            }
                            deltaBias[j][k] = -learningRate * nowD[k];
                        }
                        lastD = nowD;
                    }

                    //apluing new weights
                    for (int j = 0; j < Layers.Count; j++)
                    {
                        for (int k = 0; k < Layers[j].cells.Length; k++)
                        {
                            for (int w = 0; w < Layers[j].cells[k].weights.Length; w++)
                            {
                                Layers[j].cells[k].weights[w] += deltas[j][k][w];
                            }
                            Layers[j].cells[k].bias -= deltaBias[j][k];
                        }
                    }
                }
                epoch++;
            }
        }
        public int predict(double[] x)
        {
            double[] output = x;
            for (int i = 0; i < Layers.Count; i++)
            {
                output = Layers[i].feedForward(output);
            }
            return Array.IndexOf(output, output.Max());
        }
        public double[] feedForward(double[] x)
        {
            double[] output = x;
            for (int i = 0; i < Layers.Count; i++)
            {
                output = Layers[i].feedForward(output);
            }
            return output;
        }
        public void printWeights()
        {
            for (int i = 0; i < Layers.Count(); i++)
            {
                Console.WriteLine($"Layer {i + 1}");
                for (int j = 0; j < Layers[i].cells.Length; j++)
                {
                    Console.WriteLine($"Cell {j + 1}");
                    for (int k = 0; k < Layers[i].cells[j].weights.Length; k++)
                    {
                        Console.Write($"{Math.Round(Layers[i].cells[j].weights[k], 5)} ");
                    }
                    Console.WriteLine();
                    Console.WriteLine($"bias = {Math.Round(Layers[i].cells[j].bias, 5)}");
                }
                Console.WriteLine("======================");
            }
        }
        public (int correct, int wrong) Test(double[][] TestX, int[] TestY, bool verbose = false)
        {
            int correct = 0, wrong = 0;
            for (int i = 0; i < TestX.Length; i++)
            {
                var res = predict(TestX[i]);
                if (TestY[i] == res)
                {
                    correct++;
                    if (verbose)
                    {
                        Console.ForegroundColor = ConsoleColor.Green;
                        Console.WriteLine($"Expected - {TestY[i]}\tPredicted - {res}");
                    }
                }
                else
                {
                    wrong++;
                    if (verbose)
                    {
                        Console.ForegroundColor = ConsoleColor.DarkRed;
                        Console.WriteLine($"Expected - {TestY[i]}\tPredicted - {res}");
                    }
                }
            }
            if (verbose)
            {
                Console.ForegroundColor = ConsoleColor.Gray;
                Console.WriteLine($"Correct predictions : {correct} Wrong predictions : {wrong}\nNetwork was right at {((double)correct / (correct + wrong)) * 100}%");
            }
            return (correct, wrong);
        }
    }
}
