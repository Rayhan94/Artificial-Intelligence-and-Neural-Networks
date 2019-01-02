//rayhan miah
public class NeuralNetworkImplementation// name of program
{
	// Instance variables
	private double[] weights; // array double for weights
	private double learningRate = 1.0;// double for learning rate
	private int[][] inputs = { { 1, 0 }, { 1, 1 } };// 2d array for inputs
	private int[] outputs = { 1, 1 };// output variables

	public NeuralNetworkImplementation() {// initalise constructor
		weights = new double[9];// create new weights
		// initialise weights
		weights[0] = 0.3;
		weights[1] = 0.15;
		weights[2] = 0.15;
		weights[3] = -0.1;
		weights[4] = -0.2;
		weights[5] = 0.3;
		weights[6] = -0.1;
		weights[7] = 0.2;
		weights[8] = 0.15;
	}

	// loop for backward and forward pass takes in two parameters
	public void forwardAndBackwardPassLoop(int[][] inputs, int[] outputs) {
		double meanSquaredError = 0;
		for (int k = 0; k < inputs.length; k++)// loop through input
		{
			double actualOutput = 0;
			double outputError = 0;
			double y1, y2, y3;
			double target = outputs[k];
			int x1 = inputs[k][0];
			int x2 = inputs[k][1];

			// Forwardpass.
			// Compute outputs of the three hidden nodes
			y1 = 1 / (1 + Math.exp(-(x1 * weights[0])));
			y2 = 1 / (1 + Math.exp(-((x1 * weights[2]) + (x2 * weights[3]))));
			y3 = 1 / (1 + Math.exp(-(x2 * weights[5])));
			// Compute actual output without sigmoidal
			actualOutput = (y1 * weights[8]) + (x1 * weights[1])
					+ (y2 * weights[7]) + (x2 * weights[4]) + (y3 * weights[6]);

			// Compute output error and accumulate the mean square error for the
			// current input.
			outputError = target - actualOutput;
			meanSquaredError = meanSquaredError + Math.sqrt(outputError);
			meanSquaredError = Math.round(meanSquaredError * 10000.0) / 10000.0;

			// Backwardpass and weight update

			double[] previousWeights = new double[9];
			System.arraycopy(weights, 0, previousWeights, 0, weights.length);

			double temp = 0;
              //computing weight changes
			temp = previousWeights[8]
					+ (y1 * learningRate * outputError * actualOutput * (1 - actualOutput));
			weights[8] = Math.round(temp * 10000.0) / 10000.0;
			temp = previousWeights[1]
					+ (x1 * learningRate * outputError * actualOutput * (1 - actualOutput));
			weights[1] = Math.round(temp * 10000.0) / 10000.0;
			temp = previousWeights[7]
					+ (y2 * learningRate * outputError * actualOutput * (1 - actualOutput));
			weights[7] = Math.round(temp * 10000.0) / 10000.0;
			temp = previousWeights[4]
					+ (x2 * learningRate * outputError * actualOutput * (1 - actualOutput));
			weights[4] = Math.round(temp * 10000.0) / 10000.0;
			temp = previousWeights[6]
					+ (y3 * learningRate * outputError * actualOutput * (1 - actualOutput));
			weights[6] = Math.round(temp * 10000.0) / 10000.0;
			temp = previousWeights[0]
					+ (x1 * learningRate * outputError * y1 * (1 - y1));
			weights[0] = Math.round(temp * 10000.0) / 10000.0;
			temp = previousWeights[2]
					+ (x1 * learningRate * outputError * y2 * (1 - y2));
			weights[2] = Math.round(temp * 10000.0) / 10000.0;
			temp = previousWeights[3]
					+ (x2 * learningRate * outputError * y2 * (1 - y2));
			weights[3] = Math.round(temp * 10000.0) / 10000.0;
			temp = previousWeights[5]
					+ (x2 * learningRate * outputError * y3 * (1 - y3));
			weights[5] = Math.round(temp * 10000.0) / 10000.0;

			// Comparison of the weight updates
			for (int i = 0; i < weights.length; i++) {
				if (weights[i] == previousWeights[i]) {
					System.out.println("Weight:" + (i + 1)
							+ " remains the same");
				} else {
					System.out.println("Weight:" + (i + 1) + " changed");
				}
			}
		}
		// Print the mean squared error
		System.out.println("The mean square error is : "
				+ (meanSquaredError / inputs.length));
	}

	public void train(int epochs) {
		for (int i = 0; i < epochs; i++) {
			// do the forward and backward pass loop
			forwardAndBackwardPassLoop(inputs, outputs);
		}
	}

	// A method to demonstrate the algorithm with 100 epochs.
	public static void main(String[] args) {
		NeuralNetworkImplementation neuralNetworkImp = new NeuralNetworkImplementation();
		neuralNetworkImp.train(100);
	}
}
