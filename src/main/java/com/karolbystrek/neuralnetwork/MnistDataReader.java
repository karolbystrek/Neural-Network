package com.karolbystrek.neuralnetwork;

import java.io.DataInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class MnistDataReader {

    public static DataPoint[] readData(String dataFilePath, String labelFilePath) throws IOException {
        try (DataInputStream dataInputStream = new DataInputStream(Files.newInputStream(Path.of(dataFilePath)));
             DataInputStream labelInputStream = new DataInputStream(Files.newInputStream(Path.of(labelFilePath)))) {

            dataInputStream.readInt(); // Skip magic number.
            int numDataPoints = dataInputStream.readInt();
            int numRows = dataInputStream.readInt();
            int numCols = dataInputStream.readInt();

            labelInputStream.readInt(); // Skip magic number for labels.
            int numberOfLabels = labelInputStream.readInt();

            if (numDataPoints != numberOfLabels) {
                throw new IOException("Mismatch between data points (" + numDataPoints +
                        ") and labels (" + numberOfLabels + ").");
            }

            DataPoint[] data = new DataPoint[numDataPoints];

            for (int i = 0; i < numDataPoints; i++) {
                float[] expectedOutput = new float[10];
                float[] input = new float[numRows * numCols];

                int label = labelInputStream.readUnsignedByte();
                for (int j = 0; j < 10; j++) {
                    expectedOutput[j] = (j == label) ? 1.0f : 0.0f;
                }

                for (int r = 0; r < numRows; r++) {
                    for (int c = 0; c < numCols; c++) {
                        input[r * numCols + c] = dataInputStream.readUnsignedByte() / 255.0f;
                    }
                }
                data[i] = new DataPoint(expectedOutput, input);
            }

            return data;
        }
    }
}
