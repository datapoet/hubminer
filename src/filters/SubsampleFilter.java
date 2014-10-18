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
package filters;

import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.sparse.BOWDataSet;
import ioformat.IOARFF;
import ioformat.IOCSV;
import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import util.AuxSort;

/**
 * This filter takes a random subsample of the data.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SubsampleFilter implements FilterInterface {

    // When a number of instances is explicitly set, perc is ignored.
    private int numInst;
    private float perc;

    /**
     *
     * @param numInst Integer that is the number of instances to select.
     */
    public SubsampleFilter(int numInst) {
        this.numInst = numInst;
    }

    /**
     *
     * @param perc Float that is the percentage of instances to select.
     */
    public SubsampleFilter(float perc) {
        this.perc = perc;
        numInst = 0;
    }

    /**
     * Load, filter and save a series of datasets.
     *
     * @param inDir Input directory.
     * @param outDir Output directory.
     * @throws Exception
     */
    public void filterAndCopy(File inDir, File outDir) throws Exception {
        if (inDir == null || outDir == null) {
            return;
        }
        File[] dataFiles = inDir.listFiles();
        DataSet[] dsets = new DataSet[dataFiles.length];
        for (int i = 0; i < dsets.length; i++) {
            String dsPath = dataFiles[i].getPath();
            File dsFile = dataFiles[i];
            if (dsPath.endsWith(".csv")) {
                try {
                    IOCSV reader = new IOCSV(true, ",");
                    dsets[i] = reader.readData(dsFile);
                } catch (Exception e) {
                    IOCSV reader = new IOCSV(true, " +");
                    dsets[i] = reader.readData(dsFile);
                }
            } else if (dsPath.endsWith(".tsv")) {
                try {
                    IOCSV reader = new IOCSV(true, " +");
                    dsets[i] = reader.readData(dsFile);
                } catch (Exception e) {
                    IOCSV reader = new IOCSV(true, "\t");
                    dsets[i] = reader.readData(dsFile);
                }
            } else if (dsPath.endsWith(".arff")) {
                try {
                    IOARFF pers = new IOARFF();
                    dsets[i] = pers.loadSparse(dsPath);
                } catch (Exception e) {
                    IOARFF persister = new IOARFF();
                    dsets[i] = persister.load(dsPath);
                }
            }
        }
        // The first half of the resulting filtered array contains the selected
        // training subsamples and the second half the rejected test subsamples.
        DataSet[] filteredDatasets = filterAndCopy(dsets);
        IOARFF persister;
        for (int i = 0; i < dsets.length; i++) {
            String name = dataFiles[i].getName();
            name = name.substring(0, name.lastIndexOf("."));
            name = name + "Training";
            name = name + ".arff";
            name = outDir.getPath() + File.separator + name;
            persister = new IOARFF();
            if (filteredDatasets[i] instanceof BOWDataSet) {
                if (dsets[i].countCategories() > 1) {
                    persister.saveSparseLabeled(
                            (BOWDataSet) filteredDatasets[i], name);
                } else {
                    persister.saveSparseUnlabeled(
                            (BOWDataSet) filteredDatasets[i], name);
                }
            } else {
                if (dsets[i].countCategories() > 1) {
                    persister.saveLabeledWithIdentifiers(filteredDatasets[i],
                            name,
                            null);
                } else {
                    persister.save(filteredDatasets[i], name, null);
                }
            }
            name = dataFiles[i].getName();
            name = name.substring(0, name.lastIndexOf("."));
            name = name + "Test";
            name = name + ".arff";
            name = outDir.getPath() + File.separator + name;
            persister = new IOARFF();
            if (filteredDatasets[i + dsets.length] instanceof BOWDataSet) {
                if (dsets[i].countCategories() > 1) {
                    persister.saveSparseLabeled((BOWDataSet) filteredDatasets[i
                            + dsets.length], name);
                } else {
                    persister.saveSparseUnlabeled((BOWDataSet)
                            filteredDatasets[i + dsets.length], name);
                }
            } else {
                if (dsets[i].countCategories() > 1) {
                    persister.saveLabeledWithIdentifiers(filteredDatasets[i
                            + dsets.length], name, null);
                } else {
                    persister.save(filteredDatasets[i + dsets.length],
                            name, null);
                }
            }
        }
    }

    /**
     * Takes a random subsample from each dataset. The remaining discarded data
     * is appended in form of DataSets. It is meant for splitting data into
     * training and test sets.
     *
     * @param dsets An array of DataSet objects to filter and copy.
     * @return An array of datasets, where the first half is the selected
     * subsamples and the second half is the rejected subsamples.
     */
    public DataSet[] filterAndCopy(DataSet[] dsets) {
        if (dsets == null) {
            return null;
        }
        if (dsets.length == 0) {
            return new DataSet[0];
        }
        // An array of datasets, where the first half is the selected
        // subsamples and the second half is the rejected subsamples.
        DataSet[] results = new DataSet[dsets.length * 2];
        Random randa = new Random();
        float[] orderer = new float[dsets[0].size()];
        for (int i = 0; i < orderer.length; i++) {
            orderer[i] = randa.nextFloat();
        }
        int[] indexes = null;
        try {
            indexes = AuxSort.sortIndexedValue(orderer, true);
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
        DataInstance instance;
        for (int i = 0; i < dsets.length; i++) {
            if (dsets[i] == null) {
                continue;
            }
            if (numInst == 0) {
                numInst = (int) (perc * dsets[i].size());
            }
            int removeNumber = dsets[i].size() - numInst;
            results[i] = dsets[i].cloneDefinition();
            results[i].data = new ArrayList<>(numInst);

            for (int j = 0; j < numInst; j++) {
                instance = dsets[i].getInstance(indexes[j]);
                results[i].addDataInstance(instance);
            }
            results[i].addNewNominalAtt("original index");
            for (int j = 0; j < numInst; j++) {
                instance = results[i].getInstance(j);
                instance.sAttr[instance.sAttr.length - 1] =
                        (new Integer(indexes[j])).toString();
            }

            results[dsets.length + i] = dsets[i].cloneDefinition();
            results[dsets.length + i].data = new ArrayList<>(removeNumber);
            results[dsets.length + i].addNewNominalAtt("original index");

            for (int j = 0; j < removeNumber; j++) {
                instance = dsets[i].getInstance(indexes[dsets[i].size() - j]);
                results[dsets.length + i].addDataInstance(instance);
            }
            results[dsets.length + i].addNewNominalAtt("original index");
            for (int j = 0; j < removeNumber; j++) {
                instance = results[dsets.length + i].getInstance(j);
                instance.sAttr[instance.sAttr.length - 1] =
                        (new Integer(indexes[dsets[i].size() - j])).toString();
            }
        }
        return results;
    }

    @Override
    public DataSet filterAndCopy(DataSet dset) {
        DataSet result = null;
        try {
            result = dset.copy();
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
        filter(result);
        return result;
    }

    @Override
    public void filter(DataSet dset) {
        ShuffleFilter sf = new ShuffleFilter();
        sf.filter(dset);
        if (numInst == 0) {
            numInst = (int) (perc * dset.size());
        }
        int removeNumber = dset.size() - numInst;
        for (int i = 0; i < removeNumber; i++) {
            dset.data.remove(dset.data.size() - 1);
            dset.identifiers.data.remove(dset.identifiers.data.size() - 1);
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.out.println("This script performs random selection of"
                    + "traning and test splits for a series of datasets.");
            System.out.println("Command line parameters are as follows: ");
            System.out.println("arg0: input directory");
            System.out.println("arg1: output directory");
            System.out.println("arg2: number of instances to select for each"
                    + "training set.");
            return;
        }
        File inDir = new File(args[0]);
        File outDir = new File(args[1]);
        int numInst = Integer.parseInt(args[2]);
        SubsampleFilter sf = new SubsampleFilter(numInst);
        sf.filterAndCopy(inDir, outDir);
    }
}
