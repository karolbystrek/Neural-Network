package com.karolbystrek.neuralnetwork;

public class HiddenLayer extends Layer {

    public HiddenLayer(int numNodesIn, int numNodesOut) {
        super(numNodesIn, numNodesOut);
    }

    @Override
    public float[] forward(float[] input) {
        lastInput = input;
        float[] output = new float[numNodesOut];

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {

            lastWeightedInput[nodeOut] = biases[nodeOut];
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                lastWeightedInput[nodeOut] += weights[nodeOut][nodeIn] * input[nodeIn];
            }
            output[nodeOut] = Math.max(0.0f, lastWeightedInput[nodeOut]);
        }
        return output;
    }

    @Override
    public float[] backward(float[] outputGradient) {
        float[] inputGradient = new float[numNodesIn];
        float[] delta = new float[numNodesOut];

        for(int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            float dActivation = lastWeightedInput[nodeOut] > 0 ? 1.0f : 0.0f;
            delta[nodeOut] = outputGradient[nodeOut] * dActivation;
            biasesGradient[nodeOut] += delta[nodeOut];

            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                float inputValue = lastInput[nodeIn];
                weightsGradient[nodeOut][nodeIn] += delta[nodeOut] * inputValue;
                inputGradient[nodeIn] += weights[nodeOut][nodeIn] * delta[nodeOut];
            }
        }
        return inputGradient;
    }
}
