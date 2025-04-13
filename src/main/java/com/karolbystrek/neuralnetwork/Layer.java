package com.karolbystrek.neuralnetwork;

public abstract class Layer {

    protected int numNodesIn;
    protected int numNodesOut;

    protected float[] lastInput;
    protected float[] lastWeightedInput;

    protected float[][] weights;
    protected float[] biases;

    protected float[][] weightsGradient;
    protected float[] biasesGradient;

    protected Layer(int numNodesIn, int numNodesOut) {
        this.numNodesIn = numNodesIn;
        this.numNodesOut = numNodesOut;

        this.lastWeightedInput = new float[numNodesOut];

        this.weights = new float[numNodesOut][numNodesIn];
        this.biases = new float[numNodesOut];

        this.weightsGradient = new float[numNodesOut][numNodesIn];
        this.biasesGradient = new float[numNodesOut];

        initializeWeights();
        initializeBiases();
    }

    abstract float[] forward(float[] input);

    abstract float[] backward(float[] outputGradient);

    public void updateParameters(float learningRate) {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            biases[nodeOut] -= learningRate * biasesGradient[nodeOut];
            biasesGradient[nodeOut] = 0.0f;

            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                weights[nodeOut][nodeIn] -= learningRate * weightsGradient[nodeOut][nodeIn];
                weightsGradient[nodeOut][nodeIn] = 0.0f;
            }
        }
    }

    protected void initializeWeights() {
        float scale = (float) Math.sqrt(2.0 / numNodesIn);
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                weights[nodeOut][nodeIn] = (float) (Math.random() * 2 - 1) * scale;
                weightsGradient[nodeOut][nodeIn] = 0.0f;
            }
        }
    }

    protected void initializeBiases() {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            biases[nodeOut] = 0.0f;
            biasesGradient[nodeOut] = 0.0f;
        }
    }
}
