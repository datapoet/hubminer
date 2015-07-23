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
package statistics;

import data.representation.DataInstance;
import data.representation.util.DataMineConstants;
import draw.basic.RotatedEllipse;
import learning.unsupervised.Cluster;

/**
 * This class represents 2D data variance so that Ellipses can be drawn to
 * visualize dispersion in 2D.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Variance2D {

    // Whether to multiply the axes length by the confidence factor.
    boolean multiply = true;
    // The multiple to principal variance axes to ensure 0.95 confidence level.
    public static final double CONF95 = 2.47;
    int xIndex = 1;
    int yIndex = 0;

    public Variance2D() {
        xIndex = 1;
        yIndex = 0;
    }

    /**
     * @param multiply Boolean flag indicating whether to multiply the axes
     * length by the confidence factor.
     */
    public Variance2D(boolean multiply) {
        xIndex = 1;
        yIndex = 0;
        this.multiply = multiply;
    }

    /**
     * @param xIndex - index of the X feature in data definition.
     * @param yIndex - index of the Y feature in data definition.
     */
    public Variance2D(int xIndex, int yIndex) {
        this.xIndex = xIndex;
        this.yIndex = yIndex;
    }

    /**
     * @param multiply Boolean flag indicating whether to multiply the axes
     * @param xIndex - index of the X feature in data definition.
     * @param yIndex - index of the Y feature in data definition. length by the
     * confidence factor.
     */
    public Variance2D(boolean multiply, int xIndex, int yIndex) {
        this.xIndex = xIndex;
        this.yIndex = yIndex;
        this.multiply = multiply;
    }

    /**
     * @param clusters An array of clusters, a cluster configuration in 2D.
     * @return An array of RotatedEllipse objects corresponding to the clusters,
     * modeling their 2D variance.
     * @throws Exception
     */
    public RotatedEllipse[] findVarianceEllipseForSIFTCLusterConfiguration(
            Cluster[] clusters) throws Exception {
        if (clusters == null) {
            return null;
        }
        RotatedEllipse[] result = new RotatedEllipse[clusters.length];
        for (int i = 0; i < clusters.length; i++) {
            result[i] = findVarianceEllipseForSIFTCluster(clusters[i]);
        }
        return result;
    }

    /**
     * @param clust A cluster in 2D.
     * @return RotatedEllipse object modeling the dispersion of the cluster in
     * the plane.
     * @throws Exception
     */
    public RotatedEllipse findVarianceEllipseForSIFTCluster(Cluster clust)
            throws Exception {
        RotatedEllipse result = new RotatedEllipse();
        if (clust == null || clust.isEmpty()) {
            return null;
        }
        double m_x = 0;
        double m_y = 0;
        double s_x = 0;
        double s_y = 0;
        double s_xy = 0;
        double size = clust.size();
        DataInstance tempVect;
        for (int i = 0; i < clust.size(); i++) {
            tempVect = clust.getInstance(i);
            m_x += tempVect.fAttr[xIndex];
            m_y += tempVect.fAttr[yIndex];
        }
        m_x /= size;
        m_y /= size;
        // We have calculated the averages and now we use that to calculate the
        // variances from the covariance matrix in the x/y coordinate system.
        for (int i = 0; i < clust.size(); i++) {
            tempVect = clust.getInstance(i);
            s_x += (tempVect.fAttr[xIndex] - m_x)
                    * (tempVect.fAttr[xIndex] - m_x);
            s_y += (tempVect.fAttr[yIndex] - m_y)
                    * (tempVect.fAttr[yIndex] - m_y);
            s_xy += (tempVect.fAttr[xIndex] - m_x)
                    * (tempVect.fAttr[yIndex] - m_y);
        }
        s_x /= size;
        s_y /= size;
        s_xy /= size;
        // Now we have the elements of the covariance matrix and we plug that
        // into a formula to get the desired principal half-axes and an
        // orientation angle.
        double ro = s_xy / Math.sqrt(s_x * s_y); // Correlation coefficient.
        // If there is in fact independence, ro is vanishing and then we handle
        // that case separately.
        if (DataMineConstants.isZero(ro)) {
            result.angle = 0.;
            result.x = m_x;
            result.y = m_y;
            if (multiply) {
                result.a = Math.sqrt(s_x) * CONF95;
                result.b = Math.sqrt(s_y) * CONF95;
            } else {
                result.a = Math.sqrt(s_x);
                result.b = Math.sqrt(s_y);
            }
            return result;
        }
        // Here is the general case.
        // X axis is rotated on this given orientation angle - the length of the
        // principal axis in question.
        double p_x;
        // Y axis is rotated on this given orientation angle - the length of the
        // principal axis in question.
        double p_y;
        double angle;
        angle = 0.5 * Math.atan((2 * s_xy) / (s_x - s_y));
        double sin = Math.sin(angle);
        double cos = Math.cos(angle);
        double numerator = (s_xy * s_xy * (1. - ro * ro) / (ro * ro));
        p_x = Math.sqrt(numerator / ((s_y * cos * cos)
                - (2. * s_xy * sin * cos) + s_x * sin * sin));
        p_y = Math.sqrt(numerator / ((s_y * sin * sin)
                + (2. * s_xy * sin * cos) + s_x * cos * cos));
        result.x = m_x;
        result.y = m_y;
        if (multiply) {
            result.a = p_x * CONF95;
            result.b = p_y * CONF95;
        } else {
            result.a = p_x;
            result.b = p_y;
        }
        result.angle = angle;
        return result;
    }
}
