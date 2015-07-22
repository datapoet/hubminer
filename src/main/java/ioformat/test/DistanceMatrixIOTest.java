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
package ioformat.test;

import data.representation.util.DataMineConstants;
import ioformat.DistanceMatrixIO;
import java.io.StringReader;
import java.io.StringWriter;
import java.util.Random;
import static junit.framework.Assert.fail;
import junit.framework.TestCase;
import org.junit.Test;

/**
 * This class is a test for the usage of upper triangular distance matrix
 * persistence methods.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DistanceMatrixIOTest extends TestCase {

    /**
     * This method tests the persistence of distance matrices.
     */
    @Test
    public static void testDMatPersistence() {
        int[] dims = {1, 2, 5};
        for (int dim : dims) {
            float[][] dMat = generateToyDistanceMatrix(dim);
            try (StringWriter writer = new StringWriter();) {
                DistanceMatrixIO.printDMatToStream(dMat, writer);
                try (StringReader reader = new StringReader(
                        writer.toString())) {
                    float[][] loadedDMat = DistanceMatrixIO.loadDMatFromStream(
                            reader);
                    assertEquals(dMat.length, loadedDMat.length);
                    for (int i = 0; i < dMat.length; i++) {
                        assertEquals(dMat[i].length, loadedDMat[i].length);
                        for (int j = 0; j < dMat[i].length; j++) {
                            assertEquals(dMat[i][j], loadedDMat[i][j],
                                    DataMineConstants.EPSILON);
                        }
                    }
                } catch (Exception e) {
                    fail(e.getMessage());
                }
            } catch (Exception e) {
                fail(e.getMessage());
            }
        }
    }

    /**
     * This method generates a toy distance matrix.
     *
     * @param dim Integer representing the matrix dimensionality.
     * @return float[][] that is a random toy "distance matrix" for testing.
     */
    private static float[][] generateToyDistanceMatrix(int dim) {
        float[][] dMat = new float[dim][];
        Random randa = new Random();
        for (int i = 0; i < dim; i++) {
            dMat[i] = new float[dim - i - 1];
            for (int j = 0; j < dMat[i].length; j++) {
                dMat[i][j] = randa.nextFloat();
            }
        }
        return dMat;
    }
}
