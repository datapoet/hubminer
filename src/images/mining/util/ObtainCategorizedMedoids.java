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
package images.mining.util;

import data.representation.DataSet;
import ioformat.IOARFF;
import ioformat.images.DirectoryCategorizedDataImporter;
import java.io.File;
import learning.unsupervised.Cluster;

/**
 * A short utility script for quickly obtaining image medoids for a set of
 * categorized image files. Each class is interpreted as a cluster and the
 * medoid images are determined and their representations written to a file.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ObtainCategorizedMedoids {

    /**
     * This executes the script.
     * 
     * @param args Command line parameters, as specified.
     * @throws Exception 
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.out.println("This is a short utility script for quickly "
                    + "obtaining image medoids for a set of categorized image "
                    + "files. Each class is interpreted as a cluster and the "
                    + "medoid images are determined and their representations "
                    + "written to a file.");
            System.out.println("arg0: Image data ARFF file.");
            System.out.println("arg1: Output file path.");
            System.out.println("arg2: Directory to which the image paths are"
                    + " relative.");
            return;
        }
        DataSet imageCollectionData;
        DirectoryCategorizedDataImporter dcdi =
                new DirectoryCategorizedDataImporter(new File(args[2]));
        dcdi.importRepresentation(new File(args[0]));
        imageCollectionData = dcdi.getLoadedData();
        Cluster[] categories =
                Cluster.getClustersFromCategorizedData(imageCollectionData);
        IOARFF persister = new IOARFF();
        persister.writeMedoidsToFile(categories, imageCollectionData, args[1]);
    }
}