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
package data.generators.util;

import data.representation.DataInstance;
import data.representation.DataSet;
import java.util.ArrayList;
import java.util.Random;
import learning.supervised.Category;

/**
 * This class generates a small random test set for the clustering algorithms.
 * It creates a fixed number of time series categories based on some hard-coded
 * example time arrays. It was conceived just as a minor test, not a template
 * for more extensive tests.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class RandomizedTestDataGenerator {

    private static final float tau = 1f;
    private static final float mutProb = 0.6f;

    /**
     * Introduces a bit of noise.
     *
     * @param measArray Float value array.
     * @return
     */
    private static float[] randomize(float[] measArray) {
        Random rand = new Random();
        float[] returnArray = null;
        if (measArray != null) {
            returnArray = new float[measArray.length];
            for (int i = 0; i < measArray.length; i++) {
                if (rand.nextFloat() < mutProb) {
                    returnArray[i] = measArray[i] + tau
                            * (float) rand.nextGaussian();
                }
            }
        }
        return returnArray;
    }

    /**
     * Produces simple test data.
     *
     * @return An array of simple labeled synthetic clusters.
     */
    public static Category[] produceSimpleTestData() {
        Category[] classes = new Category[5];
        String[] names = new String[20];
        for (int i = 0; i < names.length; i++) {
            names[i] = "number " + i;
        }
        String[] IDnames = new String[1];
        IDnames[0] = "id";
        DataSet dset = new DataSet(null, names, null);
        DataSet IDdset = new DataSet(IDnames, null, null);
        ArrayList data;
        data = new ArrayList(40);
        int cnt = 0;
        DataInstance id;
        DataInstance instance;
        float[] measures = {10f, 12f, 14f, 18f, 23f, 27f, 23f, 18f, 14f, 12f,
            10f, 11f, 10f, 10.5f, 10f, 9f, 12f, 11f, 10f, 11f};
        for (int i = 0; i < 40; i++) {
            id = new DataInstance(IDdset);
            id.iAttr[0] = cnt;
            instance = new DataInstance(id, dset);
            instance.fAttr = randomize(measures);
            data.add(instance);
            cnt++;
        }
        classes[0] = new Category("class 0", dset, data);
        data = new ArrayList(30);
        float[] measures4 = {10f, 8f, 7f, 6f, 7f, 9f, 8f, 9f, 12f, 14f, 16f,
            18f, 20f, 22f, 24f, 22f, 19f, 16f, 15f, 13f};
        for (int i = 0; i < 30; i++) {
            id = new DataInstance(IDdset);
            id.iAttr[0] = cnt;
            instance = new DataInstance(id, dset);
            instance.fAttr = randomize(measures4);
            data.add(instance);
            cnt++;
        }
        classes[1] = new Category("class 1", dset, data);
        data = new ArrayList(50);
        float[] measures1 = {45f, 41f, 37f, 33f, 38f, 44f, 50f, 46f, 41f, 34f,
            38f, 45f, 51f, 44f, 37f, 42f, 46f, 41f, 36f, 32f};
        for (int i = 0; i < 50; i++) {
            id = new DataInstance(IDdset);
            id.iAttr[0] = cnt;
            instance = new DataInstance(id, dset);
            instance.fAttr = randomize(measures1);
            data.add(instance);
            cnt++;
        }
        classes[2] = new Category("class 2", dset, data);
        data = new ArrayList(60);
        float[] measures2 = {28f, 27f, 28f, 27f, 29f, 26f, 28f, 27f, 30f, 29f,
            28f, 27f, 26f, 25f, 24f, 25f, 26f, 27f, 28f, 27f};
        for (int i = 0; i < 60; i++) {
            id = new DataInstance(IDdset);
            id.iAttr[0] = cnt;
            instance = new DataInstance(id, dset);
            instance.fAttr = randomize(measures2);
            data.add(instance);
            cnt++;
        }
        classes[3] = new Category("class 3", dset, data);
        data = new ArrayList(55);
        float[] measures3 = {12f, 13f, 14f, 15f, 16f, 23f, 28f, 37f, 48f, 59f,
            51f, 43f, 38f, 32f, 23f, 19f, 17f, 11f, 12f, 10f};
        for (int i = 0; i < 55; i++) {
            id = new DataInstance(IDdset);
            id.iAttr[0] = cnt;
            instance = new DataInstance(id, dset);
            instance.fAttr = randomize(measures3);
            data.add(instance);
            cnt++;
        }
        classes[4] = new Category("class 4", dset, data);
        return classes;
    }
}