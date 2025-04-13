package com.karolbystrek.neuralnetwork;

public class DataPoint {

    private final float[] expectedOutput;
    private final float[] input;

    public DataPoint(float[] expectedOutput, float[] input) {
        this.expectedOutput = expectedOutput;
        this.input = input;
    }

    public float[] getExpectedOutput() {
        return expectedOutput;
    }

    public float[] getInput() {
        return input;
    }
}
