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
package linear;

/**
 * This class implements methods for handling linear subspaces that are spanned
 * by a set of vectors.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class LinSubspace {

    float[][] defSet = null;
    // The orthonormal basis is kept in a separate structure from the original
    // definition set.
    float[][] basis = null;
    // When used in classification, category is a class label.
    public int category;
    public int currDefSetSize = 0;

    public LinSubspace() {
    }

    /**
     * @param defSet An array of vectors that span the linear subspace.
     */
    public LinSubspace(float[][] defSet) {
        this.defSet = defSet;
    }

    /**
     * @param maxSize Maximum dimensionality of the future linear subspace.
     */
    public LinSubspace(int maxSize) {
        defSet = new float[maxSize][];
        currDefSetSize = 0;
    }

    /**
     * @return Integer that is the dimensionality of the linear subspace.
     */
    public int getDimensionality() {
        if (basis == null) {
            if (defSet == null) {
                return 0;
            } else {
                basis = BasisOrthonormalization.orthonormalize(defSet);
            }
        }
        return basis.length;
    }

    /**
     * @param vect Add a vector to the definition set of vectors spanning the
     * linear subspace.
     */
    public void addToDefSet(float[] vect) {
        // For consistency, it should only be done if it is linearly independent
        // of the current defSet.
        defSet[currDefSetSize++] = vect;
        orthonormalizeBasis();
    }

    /**
     * Remove the specified vector from the definition set.
     *
     * @param index Index of the vector to remove from the definition set of the
     * linear subspace.
     */
    public void removeFromDefSet(int index) {
        if (index < currDefSetSize && index >= 0) {
            for (int i = index + 1; i < currDefSetSize; i++) {
                defSet[i - 1] = defSet[i];
            }
            defSet[currDefSetSize - 1] =
                    new float[defSet[currDefSetSize - 1].length];
            currDefSetSize--;
        }
        orthonormalizeBasis();
    }

    /**
     * Remove a vector from the definition set that is closest to the provided
     * vector.
     *
     * @param vect Vector that is used for comparisons - to determine which
     * vector from the definition set is to be removed.
     */
    public void removeClosestToVector(float[] vect) {
        // Definition set vectors are not normalized, so it is about the cosine
        // of the angle between them.
        int closestIndex = 0;
        float closestSim = 0;
        float currSim;
        for (int i = 0; i < currDefSetSize; i++) {
            currSim = LinBasic.angleCosine(vect, defSet[i]);
            if (currSim > closestSim) {
                closestSim = currSim;
                closestIndex = i;
            }
        }
        removeFromDefSet(closestIndex);
    }

    /**
     * Orthonormalize the basis of the linear subspace.
     */
    public void orthonormalizeBasis() {
        if (defSet != null) {
            basis = BasisOrthonormalization.orthonormalize(defSet,
                    currDefSetSize);
        }
    }

    /**
     * Calculates the distance between the linear subspace and a vector as a
     * norm of the difference between the vector and its projection on the
     * subspace.
     *
     * @param vect Vector, given as a float array.
     * @return Distance between the vector and the linear subspace.
     */
    public float distanceTo(float[] vect) {
        return LinBasic.modus(LinBasic.decr(vect, projection(vect)));
    }

    /**
     * Projects the vector onto the linear subspace.
     *
     * @param vect Vector, given as a float array.
     * @return Projection of the vector onto the linear subspace.
     */
    public float[] projection(float[] vect) {
        float[] proj = new float[vect.length];
        float factor;
        if (basis == null) {
            if (defSet != null) {
                orthonormalizeBasis();
            } else {
                return null;
            }
        }
        for (int i = 0; i < basis.length; i++) {
            factor = LinBasic.dotProduct(vect, basis[i]);
            // Since the basis is also normalized.
            proj = LinBasic.add(proj,
                    LinBasic.scalarMultiply(basis[i], factor));
        }
        return proj;
    }

    /**
     * Find the closest subspace to a specified vector, among a set of specified
     * linear subspaces.
     *
     * @param subspaces An array of linear subspaces.
     * @param vect Vector, given as a float array.
     * @return Index of the closest linear subspace.
     */
    public static int findClosestSubspace(LinSubspace[] subspaces,
            float[] vect) {
        int closestIndex = 0;
        float closestSim = 0;
        float currSim;
        for (int i = 0; i < subspaces.length; i++) {
            if (subspaces[i].getDimensionality() > 0) {
                currSim = LinBasic.modus(subspaces[i].projection(vect));
            } else {
                currSim = -Float.MAX_VALUE;
            }
            if (currSim > closestSim) {
                closestSim = currSim;
                closestIndex = i;
            }
        }
        return closestIndex;
    }

    /**
     * Rotate the linear subspace.
     *
     * @param rMatrix Rotation matrix.
     */
    public void rotate(float[][] rMatrix) {
        float[][] newDefSet = new float[defSet.length][rMatrix.length];
        float[][] newBasis = new float[basis.length][rMatrix.length];
        for (int i = 0; i < currDefSetSize; i++) {
            for (int j = 0; j < rMatrix.length; j++) {
                for (int k = 0; k < defSet[i].length; k++) {
                    newDefSet[i][j] += rMatrix[j][k] * defSet[i][k];
                }
            }
        }
        for (int i = 0; i < basis.length; i++) {
            for (int j = 0; j < rMatrix.length; j++) {
                for (int k = 0; k < basis[i].length; k++) {
                    newBasis[i][j] += rMatrix[j][k] * basis[i][k];
                }
            }
        }
        basis = newBasis;
        defSet = newDefSet;
    }

    /**
     * Applies a linear transformation to the subspace.
     *
     * @param rMatrix Transformation matrix.
     */
    public void applyTransform(float[][] rMatrix) {
        // The method doesn't assume that lengths are preserved.
        float[][] newDefSet = new float[defSet.length][rMatrix.length];
        for (int i = 0; i < currDefSetSize; i++) {
            for (int j = 0; j < rMatrix.length; j++) {
                for (int k = 0; k < defSet[i].length; k++) {
                    newDefSet[i][j] += rMatrix[j][k] * defSet[i][k];
                }
            }
        }
        defSet = newDefSet;
        orthonormalizeBasis();
    }

    /**
     * Rotates the linear subspace and outputs the rotated space.
     *
     * @param rMatrix Rotation matrix.
     * @return Rotated subspace.
     */
    public LinSubspace getRotatedSpace(float[][] rMatrix) {
        float[][] newDefSet = new float[defSet.length][rMatrix.length];
        float[][] newBasis = new float[basis.length][rMatrix.length];
        for (int i = 0; i < currDefSetSize; i++) {
            for (int j = 0; j < rMatrix.length; j++) {
                for (int k = 0; k < defSet[i].length; k++) {
                    newDefSet[i][j] += rMatrix[j][k] * defSet[i][k];
                }
            }
        }
        for (int i = 0; i < basis.length; i++) {
            for (int j = 0; j < rMatrix.length; j++) {
                for (int k = 0; k < basis[i].length; k++) {
                    newBasis[i][j] += rMatrix[j][k] * basis[i][k];
                }
            }
        }
        LinSubspace result = new LinSubspace();
        result.basis = newBasis;
        result.defSet = newDefSet;
        result.currDefSetSize = currDefSetSize;
        result.category = category;
        return result;
    }

    /**
     * Performs a transformation of the original linear subspace and outputs it
     * as a new subspace.
     *
     * @param rMatrix Transformation matrix.
     * @return Transformed linear subspace.
     */
    public LinSubspace getTransformedSpace(float[][] rMatrix) {
        // The method doesn't assume that lengths are preserved.
        float[][] newDefSet = new float[defSet.length][rMatrix.length];
        for (int i = 0; i < currDefSetSize; i++) {
            for (int j = 0; j < rMatrix.length; j++) {
                for (int k = 0; k < defSet[i].length; k++) {
                    newDefSet[i][j] += rMatrix[j][k] * defSet[i][k];
                }
            }
        }
        LinSubspace result = new LinSubspace();
        result.defSet = newDefSet;
        result.currDefSetSize = currDefSetSize;
        result.category = category;
        result.orthonormalizeBasis();
        return result;
    }
}
