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
package feature.correlation.test;

import feature.correlation.DistanceCorrelation;
import feature.correlation.PearsonCorrelation;
import feature.correlation.SpearmanCorrelation;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 * Utility class for debugging and testing the correlation coefficients.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class CorrelationTester {

    /**
     * Executes the script.
     * 
     * @param args One parameter, as specified.
     * @throws Exception 
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.out.println("One command line parameter: file with two "
                    + "arrays, empty separation, in two lines");
            return;
        }
        BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(new File(args[0]))));
        double[] first = null;
        double[] second = null;
        try {
            String line1 = br.readLine();
            String line2 = br.readLine();
            String[] firstItems = line1.split(" ");
            String[] secondItems = line2.split(" ");
            first = new double[firstItems.length];
            second = new double[secondItems.length];
            for (int i = 0; i < first.length; i++) {
                first[i] = Double.parseDouble(firstItems[i]);
                second[i] = Double.parseDouble(secondItems[i]);
            }
        } catch (IOException | NumberFormatException e) {
            throw e;
        } finally {
            br.close();
        }
        float corrPears = PearsonCorrelation.correlation(first, second);
        double corrDist = DistanceCorrelation.correlation(first, second);
        double corrSpear = SpearmanCorrelation.correlation(first, second);
        System.out.println("Pears:" + corrPears + " Dist:" + corrDist
                + " Spear:" + corrSpear);
    }
}
