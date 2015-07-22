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
package learning.supervised.evaluation.cv.test;

import combinatorial.Permutation;
import java.io.StringReader;
import java.io.StringWriter;
import java.util.ArrayList;
import static junit.framework.Assert.fail;
import junit.framework.TestCase;
import learning.supervised.evaluation.cv.CVFoldsIO;
import org.junit.Test;

/**
 * This class is a unit test for testing cross-validation fold persistence.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class CVFoldsIOTest extends TestCase {
    
    /**
     * This method tests whether CV fold save/load works.
     */
    @Test
    public static void testPersistenceConsistency() throws Exception {
        int[] numTimesArr = {1, 5, 10};
        int[] numFoldsArr = {1, 2, 10};
        int dataSize = 100;
        for (int numTimes: numTimesArr) {
            for (int numFolds: numFoldsArr) {
                ArrayList<Integer>[][] allFolds = generateFolds(dataSize,
                        numTimes, numFolds);
                try (StringWriter writer = new StringWriter();) {
                    CVFoldsIO.saveAllFolds(allFolds, numTimes, numFolds,
                            writer);
                    try (StringReader reader = new StringReader(
                            writer.toString())) {
                        ArrayList<Integer>[][] loadedFolds =
                                CVFoldsIO.loadAllFolds(reader);
                        assertEquals(allFolds.length, loadedFolds.length);
                        for (int i = 0; i < allFolds.length; i++) {
                            assertEquals(allFolds[i].length,
                                    loadedFolds[i].length);
                            for (int j = 0; j < allFolds[i].length; j++) {
                                if (allFolds[i][j] == null) {
                                    if (loadedFolds[i][j] == null) {
                                        continue;
                                    } else {
                                        fail("Null element saved then loaded as"
                                                + " not null.");
                                    }
                                }
                                assertEquals(allFolds[i][j].size(),
                                        loadedFolds[i][j].size());
                                for (int k = 0; k < allFolds[i][j].size();
                                        k++) {
                                    assertEquals(allFolds[i][j].get(k),
                                            loadedFolds[i][j].get(k));
                                }
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
    } 
    
    /**
     * This method randomly generates folds (without taking account any class
     * affiliation information) for testing the cross-validation protocol.
     * 
     * @param dataSize Integer that is the data size.
     * @param numTimes Integer that is the number of fold split repetitions.
     * @param numFolds Integer that is the number of folds.
     * @return ArrayList<Integer>[][] representing the folds and their
     * repetitions in the cross-validation protocol.
     * @throws Exception 
     */
    private static ArrayList<Integer>[][] generateFolds(int dataSize,
            int numTimes, int numFolds) throws Exception {
        int[] indexPermutation;
        ArrayList<Integer>[][] allFolds = new ArrayList[numTimes][numFolds];
        for (int i = 0; i < numTimes; i++) {
            indexPermutation = Permutation.obtainRandomPermutation(dataSize);
            for (int j = 0; j < numFolds; j++) {
                allFolds[i][j] = new ArrayList<>(
                        (int)(dataSize / (float)numFolds) + 10);
            }
            for (int j = 0; j < dataSize; j++) {
                allFolds[i][indexPermutation[j] % numFolds].add(j); 
            }
        }
        return allFolds;
    }
    
}
