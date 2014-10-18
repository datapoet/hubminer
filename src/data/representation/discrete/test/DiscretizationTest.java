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
package data.representation.discrete.test;

import data.generators.util.MultiGaussianMixForClusteringTesting;
import data.representation.DataSet;
import data.representation.discrete.DiscretizedDataSet;
import data.representation.discrete.tranform.EntropyMDLDiscretizer;
import ioformat.IOARFF;
import ioformat.IOARFFDiscretized;
import java.io.File;

/**
 * This class implements a script that lets users inspect how discretization
 * works and test if it operates correctly, in case new discretization methods
 * are later included into the library. It also tests save and load of
 * discretized data.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DiscretizationTest {

    /**
     * Command line parameters specification.
     */
    public static void info() {
        System.out.println("This class tests data discretization on synthetic"
                + " Gaussian data");
        System.out.println("arg0: file for the discretized data set");
        System.out.println("arg1: file for the discretized data set after "
                + "save&load");
        System.out.println("arg2: file for the original data set");
    }

    /**
     * @param args Command line parameters, as specified in info().
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            info();
        } else {
            DataSet dset;
            // First generate some random data to test the discretizers on.
            System.out.println("Generating data...");
            MultiGaussianMixForClusteringTesting generator =
                    new MultiGaussianMixForClusteringTesting(10, 30, 4000,
                    false);
            dset = generator.generateRandomCollection();
            DiscretizedDataSet dsetDisc = new DiscretizedDataSet(dset);
            EntropyMDLDiscretizer discretizer =
                    new EntropyMDLDiscretizer(dset, dsetDisc, 10);
            // Use the discretizer to discretize the data.
            System.out.println("Forming discretization intervals...");
            discretizer.discretizeAll();
            System.out.println("Discretizing collection...");
            dsetDisc.discretizeDataSet(dset);
            // Initialize IO streams and persist the datasets.
            IOARFFDiscretized dpersister = new IOARFFDiscretized();
            IOARFF persister = new IOARFF();
            System.out.println("Persisting...");
            persister.save(dset, args[2], null);
            dpersister.saveLabeled(dsetDisc, new File(args[0]),
                    args[2]);
            System.out.println("Loading...");
            dsetDisc = dpersister.loadLabeled(new File(args[0]), dset);
            System.out.println("Persisting...");
            dpersister.saveLabeled(dsetDisc, new File(args[1]),
                    args[2]);
        }
    }
}
