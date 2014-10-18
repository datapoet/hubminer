/**
* Hub Miner: a hubness-aware machine learning experimentation library.
* Copyright (C) 2014  Nenad Tomasev. Email: nenad.tomasev at gmail.com
* 
* This program is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the Free Software
* Foundation, either version 3 of the License, or (at your option) any later
* version.
* 
* This program is distributed in the hope that it will be useful, but WITHOUT
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with
* this program. If not, see <http://www.gnu.org/licenses/>.
*/
package dimensionality_reduction;

import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import ioformat.IOARFF;
import ioformat.SupervisedLoader;
import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import transformation.TransformationInterface;
import util.CommandLineParser;

/**
 * This class implements the random projections method where the data is
 * projected onto a lower-dimensional subspace through the origin via a random
 * matrix whose columns sum up to 1.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class RandomProjection implements TransformationInterface {

    private DataSet dset;
    // The number of dimensions to reduce the data to.
    private int targetDimensionality = 2;

    /**
     * The default constructor.
     */
    public RandomProjection() {
    }

    /**
     * Initialization.
     *
     * @param dset DataSet to reduce the dimensionality of.
     */
    public RandomProjection(DataSet dset) {
        this.dset = dset;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet to reduce the dimensionality of.
     * @param targetDimensionality Integer that is the number of dimensions to
     * reduce the data to.
     */
    public RandomProjection(DataSet dset, int targetDimensionality) {
        this.dset = dset;
        this.targetDimensionality = targetDimensionality;
    }

    /**
     * Set the original data to reduce dimensionality of.
     *
     * @param dset DataSet - the original data set.
     */
    @Override
    public void setData(DataSet dset) {
        this.dset = dset;
    }

    /**
     * Generates a random projection matrix.
     *
     * @param dataDimensionality Integer that is the number of dimensions in the
     * data.
     * @param targetDimensionality Integer that is the number of dimensions to
     * reduce the data to.
     * @return float[][] that is the random projection matrix.
     */
    public float[][] generateRandomProjectionMatrix(int dataDimensionality,
            int targetDimensionality) {
        float[][] projectionMatrix =
                new float[targetDimensionality][dataDimensionality];
        float[] column;
        float denominator;
        Random randa = new Random();
        // Just in case, though this should never happen (if all zeroes are
        // generated randomly for a column).
        boolean invalid;
        // Generate random columns.
        for (int i = 0; i < dataDimensionality; i++) {
            column = new float[targetDimensionality];
            invalid = true;
            do {
                denominator = 0;
                for (int j = 0; j < targetDimensionality; j++) {
                    column[j] = randa.nextFloat();
                    denominator = Math.max(denominator, column[j]);
                }
                if (denominator > 0) {
                    invalid = false;
                }
            } while (invalid);
            for (int j = 0; j < targetDimensionality; j++) {
                projectionMatrix[j][i] = column[j] / denominator;
            }
        }
        return projectionMatrix;
    }

    @Override
    public DataSet transformData() {
        // First handle the trivial cases.
        if (dset == null || dset.isEmpty()) {
            return null;
        }
        int numFAtt = dset.getNumFloatAttr();
        if (numFAtt <= targetDimensionality) {
            DataSet trivialResult;
            try {
                trivialResult = dset.copy();
                return trivialResult;
            } catch (Exception e) {
                return dset;
            }
        }
        // Generate a random projection matrix.
        float[][] projectionMatrix = generateRandomProjectionMatrix(numFAtt,
                targetDimensionality);
        DataSet projectedDSet = new DataSet();
        // Generate dummy attribute names for the resulting projection.
        String[] fAttNames = new String[targetDimensionality];
        for (int d = 0; d < targetDimensionality; d++) {
            fAttNames[d] = "fAtt" + d;
        }
        projectedDSet.fAttrNames = fAttNames;
        // Initialize the data array for the result.
        projectedDSet.data = new ArrayList<>(dset.size());
        // Initialize zeroes data instances.
        for (int i = 0; i < dset.size(); i++) {
            // This constructor creates a zeroed float array, since it is
            // specified in the projectedDSet context.
            DataInstance instance = new DataInstance(projectedDSet);
            instance.embedInDataset(projectedDSet);
            projectedDSet.addDataInstance(instance);
            instance.setCategory(dset.getLabelOf(i));
            instance.setIdentifier(dset.getInstance(i).copyIdentifier());
        }
        // Now finally perform the random projection.
        for (int i = 0; i < targetDimensionality; i++) {
            for (int j = 0; j < dset.size(); j++) {
                for (int k = 0; k < numFAtt; k++) {
                    // Handling possible missing values or incorrect entries.
                    float fVal = dset.getInstance(j).fAttr[k];
                    if (DataMineConstants.isAcceptableFloat(fVal)) {
                        projectedDSet.getInstance(j).fAttr[i] +=
                                projectionMatrix[i][k] * fVal;
                    }
                }
            }
        }
        return projectedDSet;
    }
    
    /**
     * Performs the random projection from the file specified by the user,
     * reducing it to a specified number of dimensions and persisting the
     * results to an output file.
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-inFile", "Path to the input dataset",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outFile", "Output path", CommandLineParser.STRING,
                true, false);
        clp.addParam("-dim", "Dimensionality of data projection",
                CommandLineParser.INTEGER, true, false);
        clp.parseLine(args);
        File inFile = new File((String) (clp.getParamValues("-inFile").get(0)));
        File outFile = new File((String) (clp.getParamValues(
                "-outFile").get(0)));
        int targetDim = (Integer) (clp.getParamValues("-dim").get(0));
        DataSet inputSet = SupervisedLoader.loadData(inFile.getPath(), false);
        RandomProjection rp = new RandomProjection(inputSet, targetDim);
        DataSet output = rp.transformData();
        IOARFF saver = new IOARFF();
        saver.saveLabeled(output, outFile.getPath());
    }
}
