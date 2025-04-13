package com.karolbystrek.neuralnetwork;

public class OutputLayer extends Layer {

    public OutputLayer(int numNodesIn, int numNodesOut) {
        super(numNodesIn, numNodesOut);
    }

    @Override
    public float[] forward(float[] input) {
        lastInput = input;
        float[] output = new float[numNodesOut];

        float maxLogit = Float.NEGATIVE_INFINITY;
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            lastWeightedInput[nodeOut] = biases[nodeOut];
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                lastWeightedInput[nodeOut] += weights[nodeOut][nodeIn] * input[nodeIn];
            }
            maxLogit = Math.max(maxLogit, lastWeightedInput[nodeOut]);
        }

        float sumExp = 0.0f;
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            float expLogit = (float) Math.exp(lastWeightedInput[nodeOut] - maxLogit);
            sumExp += expLogit;
            output[nodeOut] = expLogit;
        }

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            output[nodeOut] /= sumExp;
        }

        return output;
    }

    @Override
    public float[] backward(float[] outputGradient) {
        float[] inputGradient = new float[numNodesIn];
        float[] delta = new float[numNodesOut];

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            delta[nodeOut] = outputGradient[nodeOut];
            biasesGradient[nodeOut] += delta[nodeOut];

            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                weightsGradient[nodeOut][nodeIn] += delta[nodeOut] * lastInput[nodeIn];
                inputGradient[nodeIn] += weights[nodeOut][nodeIn] * delta[nodeOut];
            }
        }
        return inputGradient;
    }
}
