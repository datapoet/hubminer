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
package optimization.stochastic.fitness;

import data.representation.DataInstance;
import data.representation.images.sift.LFeatRepresentation;
import data.representation.images.sift.LFeatVector;
import images.mining.clustering.IntraImageKMeansAdapted;
import images.mining.clustering.IntraImageKMeansAdaptedScale;
import images.mining.clustering.IntraImageKMeansWeighted;
import ioformat.images.SegmentationIO;
import ioformat.images.SiftUtil;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusteringAlg;
import util.BasicMathUtil;

/**
 * Evaluates the fitness of image segmentation by the homogeneity of SIFT
 * clusters.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SIFTSegmentationHomogeneity implements FitnessEvaluator {

    private boolean useRank = false; // Use ranks instead of distances.
    private File inImagesDir;
    private File inSIFTDir;
    private File inSegmentDir;
    private String[] nameList;
    private BufferedImage[] images;
    private LFeatRepresentation[] siftReps;
    private int[][][] segmentations;
    private int[] numSegments;
    private int minClusters = 10;
    private int maxClusters = 15;
    private float avgTotalEnt;
    private float avgImgEnt = 0;
    private float[][] segmentClassDistr;
    private int[] segmentTotals;
    private int numNonEmptySegs;

    /**
     * @param inImagesDir Directory containing the target images.
     * @param inSIFTDir Directory containing the extracted SIFT descriptors.
     * @param inSegmentDir Directory containing the segmentations.
     * @param nameList List of image names.
     * @param minClusters Minimal number of clusters to try.
     * @param maxClusters Maximal number of clusters to try.
     * @param useRank Whether to use rank or distances directly.
     * @return An instance of the SIFTSegmentationHomogeneity.
     * @throws Exception
     */
    public static SIFTSegmentationHomogeneity newInstance(
            File inImagesDir,
            File inSIFTDir,
            File inSegmentDir,
            String[] nameList,
            int minClusters,
            int maxClusters,
            boolean useRank) throws Exception {
        SIFTSegmentationHomogeneity fe = new SIFTSegmentationHomogeneity(
                inImagesDir,
                inSIFTDir,
                inSegmentDir,
                nameList,
                minClusters,
                maxClusters,
                useRank);
        fe.populate();
        return fe;
    }

    /**
     * @param inImagesDir Directory containing the target images.
     * @param inSIFTDir Directory containing the extracted SIFT descriptors.
     * @param inSegmentDir Directory containing the segmentations.
     * @param nameList List of image names.
     * @param useRank Whether to use rank or distances directly.
     * @return An instance of the SIFTSegmentationHomogeneity.
     * @throws Exception
     */
    public static SIFTSegmentationHomogeneity newInstance(
            File inImagesDir,
            File inSIFTDir,
            File inSegmentDir,
            String[] nameList,
            boolean useRank) throws Exception {
        SIFTSegmentationHomogeneity fe =
                new SIFTSegmentationHomogeneity(inImagesDir, inSIFTDir,
                inSegmentDir, nameList, useRank);
        fe.populate();
        return fe;
    }

    /**
     *
     * @param inImagesDir Directory containing the target images.
     * @param inSIFTDir Directory containing the extracted SIFT descriptors.
     * @param inSegmentDir Directory containing the segmentations.
     * @param nameList List of image names.
     * @param minClusters Minimal number of clusters to try.
     * @param maxClusters Maximal number of clusters to try.
     * @param useRank Whether to use rank or distances directly.
     */
    public SIFTSegmentationHomogeneity(
            File inImagesDir,
            File inSIFTDir,
            File inSegmentDir,
            String[] nameList,
            int minClusters,
            int maxClusters,
            boolean useRank) {
        this.inImagesDir = inImagesDir;
        this.inSIFTDir = inSIFTDir;
        this.inSegmentDir = inSegmentDir;
        this.nameList = nameList;
        this.minClusters = minClusters;
        this.maxClusters = maxClusters;
        this.useRank = useRank;
    }

    /**
     * @param inImagesDir Directory containing the target images.
     * @param inSIFTDir Directory containing the extracted SIFT descriptors.
     * @param inSegmentDir Directory containing the segmentations.
     * @param nameList List of image names.
     * @param useRank Whether to use rank or distances directly.
     */
    public SIFTSegmentationHomogeneity(
            File inImagesDir,
            File inSIFTDir,
            File inSegmentDir,
            String[] nameList,
            boolean useRank) {
        this.inImagesDir = inImagesDir;
        this.inSIFTDir = inSIFTDir;
        this.inSegmentDir = inSegmentDir;
        this.nameList = nameList;
        this.minClusters = -1;
        this.maxClusters = -1;
        this.useRank = useRank;
    }

    /**
     * Get the data from the specified input directories.
     *
     * @throws Exception
     */
    private void populate() throws Exception {
        int length = nameList.length;
        File currImageFile;
        File currSIFTFile;
        File currSegmentFile;
        images = new BufferedImage[length];
        siftReps = new LFeatRepresentation[length];
        segmentations = new int[length][][];
        numSegments = new int[length];
        for (int i = 0; i < length; i++) {
            currImageFile = new File(inImagesDir, nameList[i] + ".jpg");
            currSIFTFile = new File(inSIFTDir, nameList[i] + ".key");
            currSegmentFile = new File(inSegmentDir, nameList[i] + ".txt");
            images[i] = ImageIO.read(currImageFile);
            siftReps[i] = SiftUtil.importFeaturesFromSift(currSIFTFile);
            SegmentationIO segIO = new SegmentationIO();
            segIO.read(currSegmentFile);
            segmentations[i] = segIO.getSegmentation();
            numSegments[i] = segIO.getNumSegments();
        }
    }

    @Override
    public float evaluate(Object o) {
        DataInstance original = (DataInstance) o;
        ClusteringAlg clusterer;
        float alpha = original.fAttr[0];
        float beta = original.fAttr[1];
        float gamma = original.fAttr[2];
        int x, y;
        int segmentIndex;
        avgTotalEnt = 0;
        if (minClusters != -1 && maxClusters != -1) {
            for (int numClust = minClusters; numClust <= maxClusters;
                    numClust++) {
                for (int i = 0; i < images.length; i++) {
                    System.out.println(avgImgEnt);
                    avgImgEnt = 0;
                    segmentTotals = new int[numSegments[i]];
                    if (useRank) {
                        clusterer = new IntraImageKMeansAdaptedScale(
                                images[i],
                                siftReps[i],
                                numClust,
                                alpha,
                                beta,
                                gamma);
                    } else {
                        clusterer = new IntraImageKMeansWeighted(
                                images[i],
                                siftReps[i],
                                numClust,
                                alpha,
                                beta);
                    }
                    Cluster[] clusters = null;
                    boolean ok = false;
                    while (!ok) {
                        try {
                            clusterer.cluster();
                            clusters = clusterer.getClusters();
                            ok = true;
                        } catch (Exception e) {
                        }
                    }
                    if (clusters == null) {
                        return 0;
                    }
                    segmentClassDistr =
                            new float[numSegments[i]][clusters.length];
                    for (int j = 0; j < clusters.length; j++) {
                        if (clusters[j] != null && !clusters[j].isEmpty()) {
                            for (int k = 0; k < clusters[j].size(); k++) {
                                x = (int) ((LFeatVector)
                                        (clusters[j].getInstance(k))).getX();
                                y = (int) ((LFeatVector)
                                        (clusters[j].getInstance(k))).getY();
                                y = Math.min(segmentations[i].length - 1, y);
                                x = Math.min(segmentations[i][0].length - 1, x);
                                y = Math.max(0, y);
                                x = Math.max(0, x);
                                segmentIndex = segmentations[i][y][x];
                                segmentTotals[segmentIndex]++;
                                segmentClassDistr[segmentIndex][j]++;
                            }
                        }
                    }
                    numNonEmptySegs = 0;
                    for (int j = 0; j < numSegments[i]; j++) {
                        if (segmentTotals[j] > 0) {
                            numNonEmptySegs++;
                            for (int k = 0; k < segmentClassDistr[j].length;
                                    k++) {
                                segmentClassDistr[j][k] /=
                                        (float) segmentTotals[j];
                                if (segmentClassDistr[j][k] > 0) {
                                    avgImgEnt -= segmentClassDistr[j][k]
                                            * (float) BasicMathUtil.log2(
                                            segmentClassDistr[j][k]);
                                }
                            }
                        }
                    }
                    // Now divide and get the average within-segment entropy
                    // for this image.
                    avgImgEnt /= (float) numNonEmptySegs;
                    avgTotalEnt += avgImgEnt;
                }
            }
            avgTotalEnt /= ((float) (maxClusters - minClusters + 1)
                    * (float) images.length);
            return avgTotalEnt;
        } else {
            for (int i = 0; i < images.length; i++) {
                System.out.println(avgImgEnt);
                avgImgEnt = 0;
                segmentTotals = new int[numSegments[i]];
                if (useRank) {
                    clusterer = new IntraImageKMeansAdapted(
                            images[i],
                            siftReps[i],
                            numSegments[i],
                            alpha,
                            beta);
                } else {
                    clusterer = new IntraImageKMeansWeighted(
                            images[i],
                            siftReps[i],
                            numSegments[i],
                            alpha,
                            beta);
                }
                Cluster[] clusters = null;
                boolean ok = false;
                while (!ok) {
                    try {
                        clusterer.cluster();
                        clusters = clusterer.getClusters();
                        ok = true;
                    } catch (Exception e) {
                    }
                }
                if (clusters == null) {
                    return 0;
                }
                segmentClassDistr =
                        new float[numSegments[i]][clusters.length];
                for (int j = 0; j < clusters.length; j++) {
                    if (clusters[j] != null && !clusters[j].isEmpty()) {
                        for (int k = 0; k < clusters[j].size(); k++) {
                            x = (int) ((LFeatVector)
                                    (clusters[j].getInstance(k))).getX();
                            y = (int) ((LFeatVector)
                                    (clusters[j].getInstance(k))).getY();
                            y = Math.min(segmentations[i].length - 1, y);
                            x = Math.min(segmentations[i][0].length - 1, x);
                            y = Math.max(0, y);
                            x = Math.max(0, x);
                            segmentIndex = segmentations[i][y][x];
                            segmentTotals[segmentIndex]++;
                            segmentClassDistr[segmentIndex][j]++;
                        }
                    }
                }
                numNonEmptySegs = 0;
                for (int j = 0; j < numSegments[i]; j++) {
                    if (segmentTotals[j] > 0) {
                        numNonEmptySegs++;
                        float currEnt = 0;
                        for (int k = 0; k < segmentClassDistr[j].length;
                                k++) {
                            segmentClassDistr[j][k] /=
                                    (float) segmentTotals[j];
                            if (segmentClassDistr[j][k] > 0) {
                                currEnt -= segmentClassDistr[j][k]
                                        * (float) BasicMathUtil.log2(
                                        segmentClassDistr[j][k]);
                            }
                        }
                        System.out.print("s" + j + ":" + currEnt + " ");
                        avgImgEnt += currEnt * segmentTotals[j];
                    }
                }
                System.out.println(avgImgEnt);
                int imageTotal = 0;
                for (int j = 0; j < numSegments[i]; j++) {
                    imageTotal += segmentTotals[j];
                }
                // Now divide and get the average within-segment entropy
                // for this image.
                avgImgEnt /= (float) imageTotal;
                avgTotalEnt += avgImgEnt;
            }
            avgTotalEnt /= ((float) (maxClusters - minClusters + 1)
                    * (float) images.length);
            return avgTotalEnt;
        }
    }
}
