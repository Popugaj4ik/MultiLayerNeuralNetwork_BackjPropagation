using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AI_Laba_NeuralNet
{
    class Layer
    {
        public Cell[] cells;
        public Layer(int cellsCount)
        {
            cells = new Cell[cellsCount];
        }
        public void build(int inputCount, Func<double, double> activationFunction, Func<double, double> derivitiveFunction)
        {
            for(int i = 0; i < cells.Length; i++)
            {
                cells[i] = new Cell(inputCount, activationFunction, derivitiveFunction, bias: Program.r.NextDouble());
            }
        }
        public void build(int inputCount, Func<double, double> activationFunction, Func<double, double> derivitiveFunction, double bias)
        {
            for (int i = 0; i < cells.Length; i++)
            {
                cells[i] = new Cell(inputCount, activationFunction, derivitiveFunction, bias);
            }
        }
        public void build(int inputCount, Func<double, double>[] activationFunction, Func<double, double>[] derivitiveFunction)
        {
            if (activationFunction.Length < cells.Length && derivitiveFunction.Length < cells.Length)
                throw new ArgumentOutOfRangeException($"Not enough function to activate neurons");
            for (int i = 0; i < cells.Length; i++)
            {
                cells[i] = new Cell(inputCount, activationFunction[i], derivitiveFunction[i], bias: Program.r.NextDouble());
            }
        }
        public void build(int inputCount, Func<double, double>[] activationFunction, Func<double, double>[] derivitiveFunction, double[] bias)
        {
            if (activationFunction.Length < cells.Length && derivitiveFunction.Length < cells.Length)
                throw new ArgumentOutOfRangeException($"Not enough function to activate neurons");
            for (int i = 0; i < cells.Length; i++)
            {
                cells[i] = new Cell(inputCount, activationFunction[i], derivitiveFunction[i], bias[i]);
            }
        }
        public double[] feedForward(double[] X)
        {
            var mult = new double[cells.Length];
            for(int i = 0; i < cells.Length; i++)
            {
                mult[i] = cells[i].feedForward(X);
            }
            return mult;
        }
    }
}
