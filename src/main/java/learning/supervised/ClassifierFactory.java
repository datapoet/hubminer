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
package learning.supervised;

import distances.primary.CombinedMetric;
import java.util.ArrayList;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.meta.boosting.AdaBoostM2;
import learning.supervised.meta.boosting.baselearners.DWHFNNBoostable;
import learning.supervised.meta.boosting.baselearners.HFNNBoostable;
import learning.supervised.meta.boosting.baselearners.HIKNNBoostable;
import learning.supervised.meta.boosting.baselearners.HwKNNBoostable;
import learning.supervised.methods.RSLVQ;
import learning.supervised.methods.discrete.DNaiveBayes;
import learning.supervised.methods.discrete.DOneRule;
import learning.supervised.methods.discrete.DWeightedNaiveBayes;
import learning.supervised.methods.discrete.DZeroRule;
import learning.supervised.methods.discrete.KNNNB;
import learning.supervised.methods.discrete.LWNB;
import learning.supervised.methods.discrete.trees.DCT_ID3;
import learning.supervised.methods.knn.AKNN;
import learning.supervised.methods.knn.ANHBNN;
import learning.supervised.methods.knn.CBWkNN;
import learning.supervised.methods.knn.DWHFNN;
import learning.supervised.methods.knn.DWKNN;
import learning.supervised.methods.knn.FNN;
import learning.supervised.methods.knn.HFNN;
import learning.supervised.methods.knn.HIKNN;
import learning.supervised.methods.knn.HIKNNNonDW;
import learning.supervised.methods.knn.HwKNN;
import learning.supervised.methods.knn.KNN;
import learning.supervised.methods.knn.NHBNN;
import learning.supervised.methods.knn.NWKNN;
import learning.supervised.methods.knn.RRKNN;

/**
 * This class is used for obtaining initial classifier instances for the
 * specified neighborhood size, metric and number of classes in the data.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ClassifierFactory {

    /**
     * This method generates an initial classifier instance for the specified
     * classifier name, neighborhood size, metric and number of classes.
     *
     * @param classifierName String that is the classifier name.
     * @param numCategories Integer that is the number of categories in the
     * data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     * @return ValidateableInterface object corresponding to the parameter
     * specification.
     */
    public ValidateableInterface getClassifierForName(String classifierName,
            int numCategories, CombinedMetric cmet, int k) {
        ValidateableInterface classAlg;
        if (classifierName.equalsIgnoreCase("DNaiveBayes")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.discrete.DNaiveBayes")) {
            classAlg = new DNaiveBayes();
        } else if (classifierName.equalsIgnoreCase("DWeightedNaiveBayes")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.discrete.DWeightedNaiveBayes")) {
            classAlg = new DWeightedNaiveBayes();
        } else if (classifierName.equalsIgnoreCase("KNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.KNN")) {
            classAlg = new KNN(k, cmet);
        } else if (classifierName.equalsIgnoreCase("dwKNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.DWKNN")) {
            classAlg = new DWKNN(k, cmet);
        } else if (classifierName.equalsIgnoreCase("AKNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.AKNN")) {
            classAlg = new AKNN(k, cmet, numCategories);
        } else if (classifierName.equalsIgnoreCase("NWKNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.NWKNN")) {
            classAlg = new NWKNN(k, cmet, numCategories);
        } else if (classifierName.equalsIgnoreCase("hwKNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.HwKNN")) {
            classAlg = new HwKNN(numCategories, cmet, k);
        } else if (classifierName.equalsIgnoreCase("CBWkNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.CBWkNN")) {
            classAlg = new CBWkNN(numCategories, cmet, k);
        } else if (classifierName.equalsIgnoreCase("DWHFNN")
                || classifierName.equalsIgnoreCase("DWH-FNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.DWHFNN")) {
            classAlg = new DWHFNN(k, cmet, numCategories);
            ((DWHFNN) classAlg).noRecalcs();
        } else if (classifierName.equalsIgnoreCase("hFNN")
                || classifierName.equalsIgnoreCase("h-FNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.HFNN")) {
            classAlg = new HFNN(k, cmet, numCategories);
            ((HFNN) classAlg).noRecalcs();
        } else if (classifierName.equalsIgnoreCase("B1dwh-FNN")) {
            classAlg = new AdaBoostM2(new DWHFNNBoostable(
                    k, cmet, numCategories, DWHFNNBoostable.B1));
        } else if (classifierName.equalsIgnoreCase("B2dwh-FNN")) {
            classAlg = new AdaBoostM2(new DWHFNNBoostable(
                    k, cmet, numCategories, DWHFNNBoostable.B2));
        } else if (classifierName.equalsIgnoreCase("B1h-FNN")) {
            classAlg = new AdaBoostM2(new HFNNBoostable(
                    k, cmet, numCategories, HFNNBoostable.B1));
        } else if (classifierName.equalsIgnoreCase("B2h-FNN")) {
            classAlg = new AdaBoostM2(new HFNNBoostable(
                    k, cmet, numCategories, HFNNBoostable.B2));
        } else if (classifierName.equalsIgnoreCase("B1HIKNN")) {
            classAlg = new AdaBoostM2(new HIKNNBoostable(
                    k, cmet, numCategories, HIKNNBoostable.B1));
        } else if (classifierName.equalsIgnoreCase("B2HIKNN")) {
            classAlg = new AdaBoostM2(new HIKNNBoostable(
                    k, cmet, numCategories, HIKNNBoostable.B2));
        } else if (classifierName.equalsIgnoreCase("B1HWKNN")) {
            classAlg = new AdaBoostM2(new HwKNNBoostable(
                    numCategories, cmet, k, HwKNNBoostable.B1));
        } else if (classifierName.equalsIgnoreCase("B2HWKNN")) {
            classAlg = new AdaBoostM2(new HwKNNBoostable(
                    numCategories, cmet, k, HwKNNBoostable.B2));
        } else if (classifierName.equalsIgnoreCase("NHBNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.NHBNN")) {
            classAlg = new NHBNN(k, cmet, numCategories);
            ((NHBNN) classAlg).noRecalcs();
        } else if (classifierName.equalsIgnoreCase("ANHBNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.ANHBNN")) {
            classAlg = new ANHBNN(k, cmet, numCategories);
            ((ANHBNN) classAlg).noRecalcs();
        } else if (classifierName.equalsIgnoreCase("HIKNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.HIKNN")) {
            classAlg = new HIKNN(k, cmet, numCategories);
            ((HIKNN) classAlg).noRecalcs();
        } else if (classifierName.equalsIgnoreCase("HIKNNnondw")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.HIKNNNonDw")) {
            classAlg = new HIKNNNonDW(k, cmet, numCategories);
            ((HIKNNNonDW) classAlg).noRecalcs();
        } else if (classifierName.equalsIgnoreCase("RRKNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.RRKNN")) {
            classAlg = new RRKNN(k, cmet, numCategories);
            ((RRKNN) classAlg).noRecalcs();
        } else if (classifierName.equalsIgnoreCase("RSLVQ")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.RSLVQ")) {
            classAlg = new RSLVQ(numCategories, cmet);
        } else if (classifierName.equalsIgnoreCase("FNN")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.knn.FNN")) {
            classAlg = new FNN(k, cmet, numCategories);
        } else if (classifierName.equalsIgnoreCase("KNNNB")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.discrete.KNNNB")) {
            classAlg = new KNNNB(k, cmet);
        } else if (classifierName.equalsIgnoreCase("LWNB")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.discrete.LWNB")) {
            classAlg = new LWNB(k, cmet);
        } else if (classifierName.equalsIgnoreCase("ID3")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.discrete.trees.DCT_ID3")) {
            classAlg = new DCT_ID3();
        } else if (classifierName.equalsIgnoreCase("DZeroRule")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.discrete.DZeroRule")) {
            classAlg = new DZeroRule();
        } else if (classifierName.equalsIgnoreCase("DOneRule")
                || classifierName.equalsIgnoreCase("learning.supervised."
                + "methods.discrete.DOneRule")) {
            classAlg = new DOneRule();
        } else {
            return null;
        }
        return classAlg;
    }

    /**
     * This method generates a list of all possible classifiers for the given
     * configuration parameters, which can be useful in testing.
     *
     * @param numCategories Integer that is the number of categories in the
     * data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     * @return ArrayList<ValidateableInterface> representing a list of
     * classifiers corresponding to the parameter configuration.
     */
    public static ArrayList<ValidateableInterface> getAllClassifierInstances(
            int numCategories, CombinedMetric cmet, int k) {
        ArrayList<ValidateableInterface> classifierList = new ArrayList<>(50);
        ValidateableInterface classAlg;
        classAlg = new DNaiveBayes();
        classifierList.add(classAlg);
        classAlg = new DWeightedNaiveBayes();
        classifierList.add(classAlg);
        classAlg = new KNN(k, cmet);
        classifierList.add(classAlg);
        classAlg = new DWKNN(k, cmet);
        classifierList.add(classAlg);
        classAlg = new AKNN(k, cmet, numCategories);
        classifierList.add(classAlg);
        classAlg = new NWKNN(k, cmet, numCategories);
        classifierList.add(classAlg);
        classAlg = new HwKNN(numCategories, cmet, k);
        classifierList.add(classAlg);
        classAlg = new CBWkNN(numCategories, cmet, k);
        classifierList.add(classAlg);
        classAlg = new DWHFNN(k, cmet, numCategories);
        ((DWHFNN) classAlg).noRecalcs();
        classifierList.add(classAlg);
        classAlg = new HFNN(k, cmet, numCategories);
        ((HFNN) classAlg).noRecalcs();
        classifierList.add(classAlg);
        classAlg = new AdaBoostM2(new DWHFNNBoostable(
                k, cmet, numCategories, DWHFNNBoostable.B1));
        classifierList.add(classAlg);
        classAlg = new AdaBoostM2(new DWHFNNBoostable(
                k, cmet, numCategories, DWHFNNBoostable.B2));
        classifierList.add(classAlg);
        classAlg = new AdaBoostM2(new HFNNBoostable(
                k, cmet, numCategories, HFNNBoostable.B1));
        classifierList.add(classAlg);
        classAlg = new AdaBoostM2(new HFNNBoostable(
                k, cmet, numCategories, HFNNBoostable.B2));
        classifierList.add(classAlg);
        classAlg = new AdaBoostM2(new HIKNNBoostable(
                k, cmet, numCategories, HIKNNBoostable.B1));
        classifierList.add(classAlg);
        classAlg = new AdaBoostM2(new HIKNNBoostable(
                k, cmet, numCategories, HIKNNBoostable.B2));
        classifierList.add(classAlg);
        classAlg = new AdaBoostM2(new HwKNNBoostable(
                numCategories, cmet, k, HwKNNBoostable.B1));
        classifierList.add(classAlg);
        classAlg = new AdaBoostM2(new HwKNNBoostable(
                numCategories, cmet, k, HwKNNBoostable.B2));
        classifierList.add(classAlg);
        classAlg = new NHBNN(k, cmet, numCategories);
        ((NHBNN) classAlg).noRecalcs();
        classifierList.add(classAlg);
        classAlg = new ANHBNN(k, cmet, numCategories);
        ((ANHBNN) classAlg).noRecalcs();
        classifierList.add(classAlg);
        classAlg = new HIKNN(k, cmet, numCategories);
        ((HIKNN) classAlg).noRecalcs();
        classifierList.add(classAlg);
        classAlg = new HIKNNNonDW(k, cmet, numCategories);
        ((HIKNNNonDW) classAlg).noRecalcs();
        classifierList.add(classAlg);
        classAlg = new RRKNN(k, cmet, numCategories);
        ((RRKNN) classAlg).noRecalcs();
        classifierList.add(classAlg);
        classAlg = new RSLVQ(numCategories, cmet);
        classifierList.add(classAlg);
        classAlg = new FNN(k, cmet, numCategories);
        classifierList.add(classAlg);
        classAlg = new KNNNB(k, cmet);
        classifierList.add(classAlg);
        classAlg = new LWNB(k, cmet);
        classifierList.add(classAlg);
        classAlg = new DCT_ID3();
        classifierList.add(classAlg);
        classAlg = new DZeroRule();
        classifierList.add(classAlg);
        classAlg = new DOneRule();
        classifierList.add(classAlg);
        return classifierList;
    }
    
    /**
     * This method generates a list of all possible non-discrete classifiers for
     * the given configuration parameters, which can be useful in testing.
     *
     * @param numCategories Integer that is the number of categories in the
     * data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     * @return ArrayList<Classifier> representing a list of
     * classifiers corresponding to the parameter configuration.
     */
    public static ArrayList<Classifier> getAllNonDiscClassifierInstances(
            int numCategories, CombinedMetric cmet, int k) {
        ArrayList<Classifier> classifierList = new ArrayList<>(50);
        Classifier classAlg;
        classAlg = new KNN(k, cmet);
        classifierList.add(classAlg);
        classAlg = new DWKNN(k, cmet);
        classifierList.add(classAlg);
        classAlg = new AKNN(k, cmet, numCategories);
        classifierList.add(classAlg);
        classAlg = new NWKNN(k, cmet, numCategories);
        classifierList.add(classAlg);
        classAlg = new HwKNN(numCategories, cmet, k);
        classifierList.add(classAlg);
        classAlg = new CBWkNN(numCategories, cmet, k);
        classifierList.add(classAlg);
        classAlg = new DWHFNN(k, cmet, numCategories);
        ((DWHFNN) classAlg).noRecalcs();
        classifierList.add(classAlg);
        classAlg = new HFNN(k, cmet, numCategories);
        ((HFNN) classAlg).noRecalcs();
        classifierList.add(classAlg);
        classAlg = new AdaBoostM2(new DWHFNNBoostable(
                k, cmet, numCategories, DWHFNNBoostable.B1));
        classifierList.add(classAlg);
        classAlg = new AdaBoostM2(new DWHFNNBoostable(
                k, cmet, numCategories, DWHFNNBoostable.B2));
        classifierList.add(classAlg);
        classAlg = new AdaBoostM2(new HFNNBoostable(
                k, cmet, numCategories, HFNNBoostable.B1));
        classifierList.add(classAlg);
        classAlg = new AdaBoostM2(new HFNNBoostable(
                k, cmet, numCategories, HFNNBoostable.B2));
        classifierList.add(classAlg);
        classAlg = new AdaBoostM2(new HIKNNBoostable(
                k, cmet, numCategories, HIKNNBoostable.B1));
        classifierList.add(classAlg);
        classAlg = new AdaBoostM2(new HIKNNBoostable(
                k, cmet, numCategories, HIKNNBoostable.B2));
        classifierList.add(classAlg);
        classAlg = new AdaBoostM2(new HwKNNBoostable(
                numCategories, cmet, k, HwKNNBoostable.B1));
        classifierList.add(classAlg);
        classAlg = new AdaBoostM2(new HwKNNBoostable(
                numCategories, cmet, k, HwKNNBoostable.B2));
        classifierList.add(classAlg);
        classAlg = new NHBNN(k, cmet, numCategories);
        ((NHBNN) classAlg).noRecalcs();
        classifierList.add(classAlg);
        classAlg = new ANHBNN(k, cmet, numCategories);
        ((ANHBNN) classAlg).noRecalcs();
        classifierList.add(classAlg);
        classAlg = new HIKNN(k, cmet, numCategories);
        ((HIKNN) classAlg).noRecalcs();
        classifierList.add(classAlg);
        classAlg = new HIKNNNonDW(k, cmet, numCategories);
        ((HIKNNNonDW) classAlg).noRecalcs();
        classifierList.add(classAlg);
        classAlg = new RRKNN(k, cmet, numCategories);
        ((RRKNN) classAlg).noRecalcs();
        classifierList.add(classAlg);
        classAlg = new RSLVQ(numCategories, cmet);
        classifierList.add(classAlg);
        classAlg = new FNN(k, cmet, numCategories);
        classifierList.add(classAlg);
        return classifierList;
    }
    
    /**
     * This method generates a list of all possible discrete classifiers for the
     * given configuration parameters, which can be useful in testing.
     *
     * @param numCategories Integer that is the number of categories in the
     * data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     * @return ArrayList<DiscreteClassifier> representing a list of
     * classifiers corresponding to the parameter configuration.
     */
    public static ArrayList<DiscreteClassifier> getAllDiscClassifierInstances(
            int numCategories, CombinedMetric cmet, int k) {
        ArrayList<DiscreteClassifier> classifierList = new ArrayList<>(50);
        DiscreteClassifier classAlg;
        classAlg = new DNaiveBayes();
        classifierList.add(classAlg);
        classAlg = new DWeightedNaiveBayes();
        classifierList.add(classAlg);
        classAlg = new KNNNB(k, cmet);
        classifierList.add(classAlg);
        classAlg = new LWNB(k, cmet);
        classifierList.add(classAlg);
        classAlg = new DCT_ID3();
        classifierList.add(classAlg);
        classAlg = new DZeroRule();
        classifierList.add(classAlg);
        classAlg = new DOneRule();
        classifierList.add(classAlg);
        return classifierList;
    }
}
