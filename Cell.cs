using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AI_Laba_NeuralNet
{
    public class Cell
    {
        public double[] weights;
        public Func<double, double> activationFunction;
        public Func<double, double> derivitiveFunction;
        public double bias;
        public Cell(int inputSize, Func<double, double> activationFunction, Func<double, double> derivitiveFunction, double bias = 0.5)
        {
            weights = new double[inputSize];
            for (int i = 0; i < inputSize; i++)
            {
                weights[i] = Program.r.NextDouble();
            }
            this.bias = Program.r.NextDouble();
            this.activationFunction = activationFunction;
            this.derivitiveFunction = derivitiveFunction;
        }
        public double calculate(double[] x)
        {
            var mult = new double[weights.Length];
            for (int i = 0; i < mult.Length; i++)
            {
                mult[i] = x[i] * weights[i];
            }
            return mult.Sum() + bias;
        }
        public double feedForward(double[] x) => activationFunction(calculate(x));
    }
}
