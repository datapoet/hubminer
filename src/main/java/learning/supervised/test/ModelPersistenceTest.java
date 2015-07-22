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
package learning.supervised.test;

import data.generators.util.OverlappingGaussianGenerator;
import data.representation.DataSet;
import data.representation.discrete.DiscretizedDataSet;
import data.representation.discrete.tranform.EntropyMDLDiscretizer;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.util.ArrayList;
import static junit.framework.Assert.assertEquals;
import static junit.framework.Assert.fail;
import junit.framework.TestCase;
import learning.supervised.Classifier;
import learning.supervised.ClassifierFactory;
import learning.supervised.DiscreteClassifier;
import org.junit.Test;

/**
 * This class implements the methods that test whether saving and loading
 * classifier models works for all the implemented classifiers.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ModelPersistenceTest extends TestCase {
    
    /**
     * This method tests the classifier persistence for all non-discrete
     * classifiers.
     */
    @Test
    public static void testClassifierPersistence() throws Exception {
        int numCat = 2;
        int k = 5;
        CombinedMetric cmet = CombinedMetric.EUCLIDEAN;
        DataSet dset = OverlappingGaussianGenerator.generate(5, numCat, false,
                25, 40);
        ArrayList<Classifier> clList = ClassifierFactory.
                getAllNonDiscClassifierInstances(numCat, cmet, k);
        for (Classifier clPrototype: clList) {
            clPrototype.setData(dset.data, dset);
            clPrototype.train();
            testClassifierIO(clPrototype, dset);
        }
    }
    
    /**
     * This method tests the classifier persistence for all discrete
     * classifiers.
     */
    @Test
    public static void testDiscreteClassifierPersistence() throws Exception {
        int numCat = 2;
        int k = 5;
        CombinedMetric cmet = CombinedMetric.EUCLIDEAN;
        DataSet dset = OverlappingGaussianGenerator.generate(5, numCat, false,
                25, 40);
        DiscretizedDataSet discDSet = new DiscretizedDataSet(dset);
        EntropyMDLDiscretizer discretizer = new EntropyMDLDiscretizer(
                dset, discDSet, numCat);
        discretizer.discretizeAll();
        discDSet.discretizeDataSet(dset);
        ArrayList<DiscreteClassifier> clList = ClassifierFactory.
                getAllDiscClassifierInstances(numCat, cmet, k);
        for (DiscreteClassifier clPrototype: clList) {
            clPrototype.setData(discDSet.data, discDSet);
            clPrototype.train();
            testDiscreteClassifierIO(clPrototype, discDSet);
        }
    }
    
    /**
     * This method tests whether persistence works for a particular classifier.
     * 
     * @param trainedModel Classifier to test the save/load for.
     */
    public static void testClassifierIO(final Classifier trainedModel,
            DataSet testSet)
            throws Exception {
        if (trainedModel == null || testSet == null) {
            return;
        }
        PipedInputStream inPipe = new PipedInputStream();
        final PipedOutputStream outPipe = new PipedOutputStream(inPipe);
        try {
            Thread writer = new Thread() {
                @Override
                public void run() {
                    ObjectOutputStream ous = null;
                    try {
                        ous = new ObjectOutputStream(outPipe);
                        trainedModel.save(ous);
                    } catch (Exception e) {
                    } finally {
                        try {
                            if (ous != null) {
                                ous.close();
                            }
                        } catch (Exception e) {
                        }
                    }
                }
            };
            writer.start();
            try (ObjectInputStream ins = new ObjectInputStream(inPipe)) {
                Classifier loadedModel = (Classifier)ins.readObject();
                if (loadedModel == null) {
                    fail("Failed for " + trainedModel.getName() + " " +
                        "Null classifier loaded.");
                } else {
                    for (int index = 0; index < testSet.size(); index++) {
                        float[] clProbs =
                                trainedModel.classifyProbabilistically(
                                testSet.getInstance(index));
                        float[] loadProbs =
                                loadedModel.classifyProbabilistically(
                                testSet.getInstance(index));
                        assertEquals(clProbs.length, loadProbs.length);
                        for (int cIndex = 0; cIndex < clProbs.length;
                                cIndex++) {
                            assertEquals(clProbs[cIndex], loadProbs[cIndex],
                                    DataMineConstants.EPSILON);
                        }
                    }
                }
            } catch (Exception e) {
                fail("Failed for " + trainedModel.getName() + " " +
                        e.getMessage());
            }
        } catch (Exception e) {
            fail("Failed for " + trainedModel.getName() + " " + e.getMessage());
        }
    }
    
    /**
     * This method tests whether persistence works for a particular classifier.
     * 
     * @param trainedModel DiscreteClassifier to test the save/load for.
     */
    public static void testDiscreteClassifierIO(
            final DiscreteClassifier trainedModel,
            DiscretizedDataSet discDSet) throws Exception {
        if (trainedModel == null || discDSet == null) {
            return;
        }
        PipedInputStream inPipe = new PipedInputStream();
        final PipedOutputStream outPipe = new PipedOutputStream(inPipe);
        try {
            Thread writer = new Thread() {
                @Override
                public void run() {
                    ObjectOutputStream ous = null;
                    try {
                        ous = new ObjectOutputStream(outPipe);
                        trainedModel.save(ous);
                    } catch (Exception e) {
                    } finally {
                        try {
                            if (ous != null) {
                                ous.close();
                            }
                        } catch (Exception e) {
                        }
                    }
                }
            };
            writer.start();
            try (ObjectInputStream ins = new ObjectInputStream(inPipe)) {
                DiscreteClassifier loadedModel = DiscreteClassifier.load(ins);
                if (loadedModel == null) {
                    fail("Failed for " + trainedModel.getName() + " " +
                        "Null classifier loaded.");
                } else {
                    for (int index = 0; index < discDSet.size(); index++) {
                        float[] clProbs =
                                trainedModel.classifyProbabilistically(
                                discDSet.getInstance(index));
                        float[] loadProbs =
                                loadedModel.classifyProbabilistically(
                                discDSet.getInstance(index));
                        assertEquals(clProbs.length, loadProbs.length);
                        for (int cIndex = 0; cIndex < clProbs.length;
                                cIndex++) {
                            assertEquals(clProbs[cIndex], loadProbs[cIndex],
                                    DataMineConstants.EPSILON);
                        }
                    }
                }
            } catch (Exception e) {
                fail("Failed for " + trainedModel.getName() + " " +
                        e.getMessage());
            }
        } catch (Exception e) {
            fail("Failed for " + trainedModel.getName() + " " + e.getMessage());
        }
    }
    
}
