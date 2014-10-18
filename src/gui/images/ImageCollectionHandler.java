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
package gui.images;

import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.images.color.ColorHistogramVector;
import data.representation.images.quantized.QuantizedImageDistribution;
import data.representation.images.quantized.QuantizedImageDistributionDataSet;
import data.representation.images.quantized.QuantizedImageHistogram;
import data.representation.images.quantized.QuantizedImageHistogramDataSet;
import data.representation.images.sift.SIFTRepresentation;
import distances.primary.ColorsAndCodebooksMetric;
import distances.primary.CombinedMetric;
import distances.primary.Manhattan;
import images.mining.codebook.SIFTCodeBook;
import images.mining.codebook.SIFTCodebookMaker;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.images.ConvertJPGToPGM;
import ioformat.images.SiftUtil;
import ioformat.images.ThumbnailMaker;
import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Timer;
import java.util.TimerTask;
import javax.imageio.ImageIO;
import javax.swing.ButtonGroup;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.JProgressBar;
import javax.swing.JRadioButton;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusterConfigurationCleaner;
import learning.unsupervised.ClusteringAlg;
import learning.unsupervised.evaluation.quality.OptimalConfigurationFinder;
import learning.unsupervised.methods.FastKMeans;
import util.FileCounter;

/**
 * This GUI allows the users to do some batch image processing and SIFT feature
 * extraction via SiftWin. More types of features should be included in future
 * versions. It can also cluster images and show a summary of clustering
 * results, as well as generate quantized feature representations via feature
 * clustering and codebook assignments. Apart from SIFT features, this class can
 * also extract the color histogram information from images, combine the two
 * representations into one - and make image thumbnails for later visualization.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ImageCollectionHandler extends javax.swing.JFrame {

    private File currentDirectory = new File(".");
    // Number of images in the dataset.
    private int numImages = 0;
    // Progress of the current task, as a percentage.
    private int currProgressPercentage = 0;
    private int secondsElapsed = 0;
    // Directory that contains the image subdirectories.
    private File imagesDirectory = null;
    // Workspace directory.
    private File workspace = null;
    // File containing the codebook definitions.
    private File codebookFile = null;
    // File containing the codebook occurrence profiles.
    private File codebookProfileFile = null;
    // File for the clustering results when clustering is performed on the
    // images.
    private File clustersFile = null;
    // File containing the SIFT features extracted from the images.
    private File siftDirectory = null;
    // Directories and files for the quantized image histogram representations,
    // as well as color histograms and the combined representation that consists
    // of both feature types.
    private File siftHistogramDirectory = null;
    private File siftHistogramFile = null;
    private File siftHistogramFileNW = null;
    private File colorHistogramDirectory = null;
    private File colorHistogramFile = null;
    private File combinedFile = null;
    // Temporary PGM directory and file, which are being generated prior to
    // SIFT feature extraction via SiftWin.
    private File pgmDir = null;
    private File pgmFile = null;
    // Directory for the generated thumbnail images.
    private File thumbnailsDirectory = null;
    // The combined, final, data representation.
    private DataSet combinedRepresentation = null;
    // Color histogram image representation.
    private DataSet colorHistogramRepresentation = null;
    // Object containing the codebook definitions.
    private SIFTCodeBook codebook = null;
    // Object containing the logic for codebook calculations.
    private SIFTCodebookMaker cbm = null;
    // Normalized and non-normalized SIFT quantized feature representations.
    private QuantizedImageDistributionDataSet siftDistributions = null;
    private QuantizedImageHistogramDataSet siftBOW = null;
    // Image clusters, if clustering has been performed.
    private Cluster[] clusters = null;
    // Cluster representatives - closest images to cluster centroids.
    private DataInstance[][] clusterRepresentatives = null;
    private BufferedImage[][] clusterRepresentativesImages = null;
    // Timer objects for calculating the duration of certain tasks.
    private Timer timer;
    private javax.swing.Timer timeTimer;
    // Relative paths of all the images in the data to their parent directory.
    private ArrayList<String> imageRelativePaths = null;
    // Whether the entire directory structure that corresponds to the original
    // directory structure of the images directory has been generated in all the
    // target output and temporary directories that this tool uses.
    private boolean dirStructureCreated = false;
    // Minimal and maximal cluster numbers to try when performing clustering.
    private int minClust = -1;
    private int maxClust = -1;
    // Index of the clustering quality index to use for finding the optimal
    // cluster configuration.
    int validityIndex = 2;
    private static final int DUNN = 0;
    private static final int DAVIES_BOULDIN = 1;
    private static final int SILHOUETTE = 2;

    /**
     * Creates new form ImageCollectionHandler
     */
    public ImageCollectionHandler() {
        initComponents();
        initialization();
    }

    /**
     * Initialization.
     */
    private void initialization() {
        this.setTitle("Image collection handler");
        progressBar.setMaximum(100);
        progressBar.setMinimum(0);
        progressBar.setValue(0);
        progressBar.setStringPainted(true);
        progressBar.setOrientation(JProgressBar.HORIZONTAL);
        progressBar.setForeground(Color.ORANGE);
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        validityButtonGroup = new javax.swing.ButtonGroup();
        collectionDirLabel = new javax.swing.JLabel();
        collectionDirValueLabel = new javax.swing.JLabel();
        collectionSizeLabel = new javax.swing.JLabel();
        collectionSizeValueLabel = new javax.swing.JLabel();
        workspaceLabel = new javax.swing.JLabel();
        workspaceValueLabel = new javax.swing.JLabel();
        codeBookLabel = new javax.swing.JLabel();
        codeBookValueLabel = new javax.swing.JLabel();
        codeBookSizeLabel = new javax.swing.JLabel();
        codeBookSizeValueLabel = new javax.swing.JLabel();
        clusterFileLabel = new javax.swing.JLabel();
        clusterFileValueLabel = new javax.swing.JLabel();
        clusteringIntervalLabel = new javax.swing.JLabel();
        minClustTextField = new javax.swing.JTextField();
        maxClustTextField = new javax.swing.JTextField();
        confirmClustNumButton = new javax.swing.JButton();
        validityChoiceLabel = new javax.swing.JLabel();
        dunnValidityRadio = new javax.swing.JRadioButton();
        dbValidityRadio = new javax.swing.JRadioButton();
        silhouetteValidityRadio = new javax.swing.JRadioButton();
        statusLabel = new javax.swing.JLabel();
        statusValueLabel = new javax.swing.JLabel();
        progressBar = new javax.swing.JProgressBar();
        resultingNumLabel = new javax.swing.JLabel();
        resultingNumValueLabel = new javax.swing.JLabel();
        inspectLabel = new javax.swing.JLabel();
        inspectionNumberField = new javax.swing.JTextField();
        visualizeButton = new javax.swing.JButton();
        displayedClusterNumberLabel = new javax.swing.JLabel();
        displayedClusterNumberValueLabel = new javax.swing.JLabel();
        medoidLabel = new javax.swing.JLabel();
        medoidValueLabel = new javax.swing.JLabel();
        displayedClusterElementNumberLabel = new javax.swing.JLabel();
        displayedClusterElementNumberValueLabel = new javax.swing.JLabel();
        representativeImagesLabel = new javax.swing.JLabel();
        imagePanel1 = new gui.images.ImagePanel();
        imagePanel2 = new gui.images.ImagePanel();
        imagePanel3 = new gui.images.ImagePanel();
        medoidPathLabel = new javax.swing.JLabel();
        secondPathLabel = new javax.swing.JLabel();
        thirdPathLabel = new javax.swing.JLabel();
        reinitButton = new javax.swing.JButton();
        executionLabel = new javax.swing.JLabel();
        executionValueLabel = new javax.swing.JLabel();
        jMenuBar1 = new javax.swing.JMenuBar();
        collectionMenu = new javax.swing.JMenu();
        collectionMenu.setMnemonic(KeyEvent.VK_C);
        setCollectionItem = new javax.swing.JMenuItem();
        chooseWorkspaceItem = new javax.swing.JMenuItem();
        prepareCollectionSubMenu = new javax.swing.JMenu();
        siftItem = new javax.swing.JMenuItem();
        colorsItem = new javax.swing.JMenuItem();
        extractSIFTHistogramsItem = new javax.swing.JMenuItem();
        extractSIFTHistogramsItem.setMnemonic(KeyEvent.VK_M);
        loadFromWorkSpaceItem = new javax.swing.JMenuItem();
        loadFromWorkSpaceItem.setMnemonic(KeyEvent.VK_L);
        combinedRepresentationItem = new javax.swing.JMenu();
        combinedRepresentationItem.setMnemonic(KeyEvent.VK_C);
        makeCombinedItem = new javax.swing.JMenuItem();
        makeCombinedItem.setMnemonic(KeyEvent.VK_M);
        loadCombinedItem = new javax.swing.JMenuItem();
        loadCombinedItem.setMnemonic(KeyEvent.VK_L);
        saveCombinedItem = new javax.swing.JMenuItem();
        saveCombinedItem.setMnemonic(KeyEvent.VK_S);
        makeThumbnailsItem = new javax.swing.JMenuItem();
        makeThumbnailsItem.setMnemonic(KeyEvent.VK_T);
        codeBookMenu = new javax.swing.JMenu();
        codeBookMenu.setMnemonic(KeyEvent.VK_B);
        loadCodeBookItem = new javax.swing.JMenuItem();
        calculateCodeBookItem = new javax.swing.JMenuItem();
        codebookProfileMenuItem = new javax.swing.JMenuItem();
        clusteringMenu = new javax.swing.JMenu();
        clusteringMenu.setMnemonic(KeyEvent.VK_L);
        clusterImagesItem = new javax.swing.JMenuItem();
        saveClustersItem = new javax.swing.JMenuItem();
        saveClustersItem.setMnemonic(KeyEvent.VK_S);
        loadClustersItem = new javax.swing.JMenuItem();
        loadClustersItem.setMnemonic(KeyEvent.VK_L);
        visualizeItem = new javax.swing.JMenuItem();
        infoMenu = new javax.swing.JMenu();
        infoMenu.setMnemonic(KeyEvent.VK_I);
        aboutItem = new javax.swing.JMenuItem();
        aboutItem.setMnemonic(KeyEvent.VK_A);

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        collectionDirLabel.setText("Collection:");

        collectionDirValueLabel.setText("-");

        collectionSizeLabel.setText("Num Images:");

        collectionSizeValueLabel.setText("-");

        workspaceLabel.setText("Workspace:");

        workspaceValueLabel.setText("-");

        codeBookLabel.setText("Codebook:");

        codeBookValueLabel.setText("-");

        codeBookSizeLabel.setText("Codebook size:");

        codeBookSizeValueLabel.setText("-");

        clusterFileLabel.setText("File with clusters:");

        clusterFileValueLabel.setText("-");

        clusteringIntervalLabel.setText("Choose clustering interval:");

        minClustTextField.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                minClustTextFieldActionPerformed(evt);
            }
        });

        maxClustTextField.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                maxClustTextFieldActionPerformed(evt);
            }
        });

        confirmClustNumButton.setText("Confirm");
        confirmClustNumButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                confirmClustNumButtonActionPerformed(evt);
            }
        });

        validityChoiceLabel.setText("Choose validity measure:");

        validityButtonGroup.add(dunnValidityRadio);
        dunnValidityRadio.setText("Dunn Index");

        validityButtonGroup.add(dbValidityRadio);
        dbValidityRadio.setText("Davies Boulding Index");

        validityButtonGroup.add(silhouetteValidityRadio);
        silhouetteValidityRadio.setText("Silhouette Index");

        statusLabel.setText("Status:");

        statusValueLabel.setText("-");

        resultingNumLabel.setText("Resulting number of clusters:");

        resultingNumValueLabel.setText("-");

        inspectLabel.setText("Inspect cluster (choose number):");

        visualizeButton.setText("Go!");
        visualizeButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                visualizeButtonActionPerformed(evt);
            }
        });

        displayedClusterNumberLabel.setText("Cluster number:");

        displayedClusterNumberValueLabel.setText("-");

        medoidLabel.setText("Medoid: ");

        medoidValueLabel.setText("-");

        displayedClusterElementNumberLabel.setText("Number of Elements:");

        displayedClusterElementNumberValueLabel.setText("-");

        representativeImagesLabel.setText("Representative images from selected cluster: ");

        org.jdesktop.layout.GroupLayout imagePanel1Layout = new org.jdesktop.layout.GroupLayout(imagePanel1);
        imagePanel1.setLayout(imagePanel1Layout);
        imagePanel1Layout.setHorizontalGroup(
            imagePanel1Layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
            .add(0, 255, Short.MAX_VALUE)
        );
        imagePanel1Layout.setVerticalGroup(
            imagePanel1Layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
            .add(0, 127, Short.MAX_VALUE)
        );

        org.jdesktop.layout.GroupLayout imagePanel2Layout = new org.jdesktop.layout.GroupLayout(imagePanel2);
        imagePanel2.setLayout(imagePanel2Layout);
        imagePanel2Layout.setHorizontalGroup(
            imagePanel2Layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
            .add(0, 170, Short.MAX_VALUE)
        );
        imagePanel2Layout.setVerticalGroup(
            imagePanel2Layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
            .add(0, 89, Short.MAX_VALUE)
        );

        org.jdesktop.layout.GroupLayout imagePanel3Layout = new org.jdesktop.layout.GroupLayout(imagePanel3);
        imagePanel3.setLayout(imagePanel3Layout);
        imagePanel3Layout.setHorizontalGroup(
            imagePanel3Layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
            .add(0, 170, Short.MAX_VALUE)
        );
        imagePanel3Layout.setVerticalGroup(
            imagePanel3Layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
            .add(0, 94, Short.MAX_VALUE)
        );

        medoidPathLabel.setText("---------------------------------------------------");

        secondPathLabel.setText("----------------------------------");

        thirdPathLabel.setText("----------------------------------");

        reinitButton.setText("Reinit");
        reinitButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                reinitButtonActionPerformed(evt);
            }
        });

        executionLabel.setText("Execution:");

        executionValueLabel.setText("-");

        collectionMenu.setText("<html><u>C</u>ollection");

        setCollectionItem.setText("<html><u>S</u>et Collection");
        setCollectionItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                setCollectionItemActionPerformed(evt);
            }
        });
        collectionMenu.add(setCollectionItem);

        chooseWorkspaceItem.setText("<html>Choose <u>W</u>orkspace directory");
        chooseWorkspaceItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                chooseWorkspaceItemActionPerformed(evt);
            }
        });
        collectionMenu.add(chooseWorkspaceItem);

        prepareCollectionSubMenu.setText("<html><u>P</u>repare collection");

        siftItem.setText("<html>Extract <u>S</u>IFT features");
        siftItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                siftItemActionPerformed(evt);
            }
        });
        prepareCollectionSubMenu.add(siftItem);

        colorsItem.setText("<html>Extract <u>C</u>olor histograms");
        colorsItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                colorsItemActionPerformed(evt);
            }
        });
        prepareCollectionSubMenu.add(colorsItem);

        extractSIFTHistogramsItem.setText("<html><u>M</u>ake codebook SIFT representation");
        extractSIFTHistogramsItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                extractSIFTHistogramsItemActionPerformed(evt);
            }
        });
        prepareCollectionSubMenu.add(extractSIFTHistogramsItem);

        collectionMenu.add(prepareCollectionSubMenu);

        loadFromWorkSpaceItem.setText("<html><u>L</u>oad data from Workspace");
        loadFromWorkSpaceItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                loadFromWorkSpaceItemActionPerformed(evt);
            }
        });
        collectionMenu.add(loadFromWorkSpaceItem);

        combinedRepresentationItem.setText("<html><u>C</u>ombined representation");

        makeCombinedItem.setText("<html><u>M</u>ake");
        makeCombinedItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                makeCombinedItemActionPerformed(evt);
            }
        });
        combinedRepresentationItem.add(makeCombinedItem);

        loadCombinedItem.setText("<html><u>L</u>oad");
        loadCombinedItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                loadCombinedItemActionPerformed(evt);
            }
        });
        combinedRepresentationItem.add(loadCombinedItem);

        saveCombinedItem.setText("<html><u>S</u>ave");
        saveCombinedItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                saveCombinedItemActionPerformed(evt);
            }
        });
        combinedRepresentationItem.add(saveCombinedItem);

        collectionMenu.add(combinedRepresentationItem);

        makeThumbnailsItem.setText("<html>Make <u>T</u>humbnails");
        makeThumbnailsItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                makeThumbnailsItemActionPerformed(evt);
            }
        });
        collectionMenu.add(makeThumbnailsItem);

        jMenuBar1.add(collectionMenu);

        codeBookMenu.setText("<html>Code <u>B</u>ook");

        loadCodeBookItem.setText("<html><u>L</u>oad codebook");
        loadCodeBookItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                loadCodeBookItemActionPerformed(evt);
            }
        });
        codeBookMenu.add(loadCodeBookItem);

        calculateCodeBookItem.setText("<html><u>C</u>alculate from collection");
        calculateCodeBookItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                calculateCodeBookItemActionPerformed(evt);
            }
        });
        codeBookMenu.add(calculateCodeBookItem);

        codebookProfileMenuItem.setText("Calculate and Save Codebook Profile");
        codebookProfileMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                codebookProfileMenuItemActionPerformed(evt);
            }
        });
        codeBookMenu.add(codebookProfileMenuItem);

        jMenuBar1.add(codeBookMenu);

        clusteringMenu.setText("<html>C<u>l</u>ustering");

        clusterImagesItem.setText("<html><u>C</u>luster images");
        clusterImagesItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                clusterImagesItemActionPerformed(evt);
            }
        });
        clusteringMenu.add(clusterImagesItem);

        saveClustersItem.setText("<html><u>S</u>ave clusters");
        saveClustersItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                saveClustersItemActionPerformed(evt);
            }
        });
        clusteringMenu.add(saveClustersItem);

        loadClustersItem.setText("<html><u>L</u>oad already calculated clusters");
        loadClustersItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                loadClustersItemActionPerformed(evt);
            }
        });
        clusteringMenu.add(loadClustersItem);

        visualizeItem.setText("<html><u>V</u>isualize clusters");
        clusteringMenu.add(visualizeItem);

        jMenuBar1.add(clusteringMenu);

        infoMenu.setText("<html><u>I</u>nfo");

        aboutItem.setText("<html><u>A</u>bout");
        aboutItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                aboutItemActionPerformed(evt);
            }
        });
        infoMenu.add(aboutItem);

        jMenuBar1.add(infoMenu);

        setJMenuBar(jMenuBar1);

        org.jdesktop.layout.GroupLayout layout = new org.jdesktop.layout.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
            .add(layout.createSequentialGroup()
                .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
                    .add(layout.createSequentialGroup()
                        .addContainerGap()
                        .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
                            .add(layout.createSequentialGroup()
                                .add(clusterFileLabel)
                                .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                                .add(clusterFileValueLabel, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 318, Short.MAX_VALUE))
                            .add(layout.createSequentialGroup()
                                .add(codeBookSizeLabel)
                                .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                                .add(codeBookSizeValueLabel, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 328, Short.MAX_VALUE))
                            .add(layout.createSequentialGroup()
                                .add(codeBookLabel)
                                .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                                .add(codeBookValueLabel, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 349, Short.MAX_VALUE))
                            .add(layout.createSequentialGroup()
                                .add(workspaceLabel)
                                .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                                .add(workspaceValueLabel, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 344, Short.MAX_VALUE))
                            .add(layout.createSequentialGroup()
                                .add(collectionDirLabel)
                                .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                                .add(collectionDirValueLabel, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 351, Short.MAX_VALUE))
                            .add(layout.createSequentialGroup()
                                .add(collectionSizeLabel)
                                .addPreferredGap(org.jdesktop.layout.LayoutStyle.UNRELATED)
                                .add(collectionSizeValueLabel, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 334, Short.MAX_VALUE)))
                        .add(110, 110, 110))
                    .add(layout.createSequentialGroup()
                        .add(28, 28, 28)
                        .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
                            .add(layout.createSequentialGroup()
                                .add(clusteringIntervalLabel)
                                .add(18, 18, 18)
                                .add(minClustTextField, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 45, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                                .add(18, 18, 18)
                                .add(maxClustTextField, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 46, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                                .add(18, 18, 18)
                                .add(confirmClustNumButton))
                            .add(layout.createSequentialGroup()
                                .add(validityChoiceLabel)
                                .add(18, 18, 18)
                                .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
                                    .add(dbValidityRadio)
                                    .add(dunnValidityRadio)
                                    .add(silhouetteValidityRadio)))
                            .add(layout.createSequentialGroup()
                                .add(statusLabel)
                                .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                                .add(statusValueLabel, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 336, Short.MAX_VALUE))
                            .add(progressBar, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 377, Short.MAX_VALUE)
                            .add(layout.createSequentialGroup()
                                .add(resultingNumLabel)
                                .addPreferredGap(org.jdesktop.layout.LayoutStyle.UNRELATED)
                                .add(resultingNumValueLabel, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 227, Short.MAX_VALUE))
                            .add(layout.createSequentialGroup()
                                .add(inspectLabel)
                                .addPreferredGap(org.jdesktop.layout.LayoutStyle.UNRELATED)
                                .add(inspectionNumberField, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 47, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                                .add(18, 18, 18)
                                .add(visualizeButton))
                            .add(layout.createSequentialGroup()
                                .add(executionLabel)
                                .addPreferredGap(org.jdesktop.layout.LayoutStyle.UNRELATED)
                                .add(executionValueLabel, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 316, Short.MAX_VALUE)))
                        .add(18, 18, 18)
                        .add(reinitButton, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 98, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)))
                .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
                    .add(layout.createSequentialGroup()
                        .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
                            .add(layout.createSequentialGroup()
                                .add(displayedClusterNumberLabel)
                                .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                                .add(displayedClusterNumberValueLabel, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 487, Short.MAX_VALUE))
                            .add(layout.createSequentialGroup()
                                .add(medoidLabel)
                                .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                                .add(medoidValueLabel, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 439, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)))
                        .add(6, 6, 6))
                    .add(layout.createSequentialGroup()
                        .add(displayedClusterElementNumberLabel)
                        .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                        .add(displayedClusterElementNumberValueLabel, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 470, Short.MAX_VALUE))
                    .add(representativeImagesLabel)
                    .add(layout.createSequentialGroup()
                        .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING, false)
                            .add(medoidPathLabel, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .add(imagePanel1, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                        .add(75, 75, 75)
                        .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING, false)
                            .add(secondPathLabel, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .add(imagePanel3, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .add(imagePanel2, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .add(thirdPathLabel, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
            .add(layout.createSequentialGroup()
                .add(24, 24, 24)
                .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.BASELINE)
                    .add(collectionDirLabel)
                    .add(collectionDirValueLabel)
                    .add(displayedClusterNumberLabel)
                    .add(displayedClusterNumberValueLabel))
                .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.BASELINE)
                    .add(collectionSizeLabel)
                    .add(collectionSizeValueLabel)
                    .add(medoidLabel)
                    .add(medoidValueLabel))
                .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.BASELINE)
                    .add(workspaceLabel)
                    .add(workspaceValueLabel)
                    .add(displayedClusterElementNumberLabel)
                    .add(displayedClusterElementNumberValueLabel))
                .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
                    .add(layout.createSequentialGroup()
                        .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                        .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.BASELINE)
                            .add(codeBookLabel)
                            .add(codeBookValueLabel))
                        .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                        .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.BASELINE)
                            .add(codeBookSizeLabel)
                            .add(codeBookSizeValueLabel))
                        .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                        .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.BASELINE)
                            .add(clusterFileLabel)
                            .add(clusterFileValueLabel)
                            .add(representativeImagesLabel))
                        .add(30, 30, 30)
                        .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.BASELINE)
                            .add(clusteringIntervalLabel)
                            .add(minClustTextField, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                            .add(maxClustTextField, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                            .add(confirmClustNumButton))
                        .add(31, 31, 31)
                        .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.BASELINE)
                            .add(validityChoiceLabel)
                            .add(dunnValidityRadio))
                        .addPreferredGap(org.jdesktop.layout.LayoutStyle.UNRELATED)
                        .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING, false)
                            .add(layout.createSequentialGroup()
                                .add(dbValidityRadio)
                                .addPreferredGap(org.jdesktop.layout.LayoutStyle.UNRELATED)
                                .add(silhouetteValidityRadio)
                                .add(14, 14, 14)
                                .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.BASELINE)
                                    .add(resultingNumLabel)
                                    .add(resultingNumValueLabel))
                                .add(39, 39, 39)
                                .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.BASELINE)
                                    .add(inspectLabel)
                                    .add(inspectionNumberField, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                                    .add(visualizeButton))
                                .add(42, 42, 42)
                                .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.BASELINE)
                                    .add(executionLabel)
                                    .add(executionValueLabel))
                                .addPreferredGap(org.jdesktop.layout.LayoutStyle.UNRELATED)
                                .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.BASELINE)
                                    .add(statusLabel)
                                    .add(statusValueLabel))
                                .add(18, 18, 18)
                                .add(progressBar, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE))
                            .add(layout.createSequentialGroup()
                                .add(imagePanel1, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(org.jdesktop.layout.LayoutStyle.UNRELATED)
                                .add(medoidPathLabel)
                                .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .add(reinitButton, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 39, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE))))
                    .add(layout.createSequentialGroup()
                        .add(121, 121, 121)
                        .add(imagePanel2, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(org.jdesktop.layout.LayoutStyle.UNRELATED)
                        .add(secondPathLabel)
                        .add(32, 32, 32)
                        .add(imagePanel3, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(org.jdesktop.layout.LayoutStyle.UNRELATED)
                        .add(thirdPathLabel)))
                .addContainerGap(29, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    /**
     * The minimal and maximal cluster values are only taken when the run has
     * been initiated, they are not being read incrementally from the text
     * fields.
     *
     * @param evt ActionEvent object.
     */
    private void maxClustTextFieldActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_maxClustTextFieldActionPerformed
    }//GEN-LAST:event_maxClustTextFieldActionPerformed

    /**
     * The minimal and maximal cluster values are only taken when the run has
     * been initiated, they are not being read incrementally from the text
     * fields.
     *
     * @param evt ActionEvent object.
     */
    private void minClustTextFieldActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_minClustTextFieldActionPerformed
    }//GEN-LAST:event_minClustTextFieldActionPerformed

    /**
     * Load the image data from the specified directory.
     *
     * @param evt ActionEvent object.
     */
    private void setCollectionItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_setCollectionItemActionPerformed
        JFileChooser jfc = new JFileChooser(currentDirectory);
        jfc.setDialogTitle("Select the directory containing all the images");
        jfc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        int rVal = jfc.showOpenDialog(ImageCollectionHandler.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentDirectory = jfc.getSelectedFile();
            try {
                imagesDirectory = currentDirectory;
                collectionDirValueLabel.setText(imagesDirectory.getPath());
                FileCounter fc = new FileCounter(imagesDirectory, "jpg");
                numImages = fc.countFiles();
                imageRelativePaths = fc.findAllRelativePaths();
                collectionSizeValueLabel.setText(Integer.toString(numImages));
                statusValueLabel.setText("Data directory specified");
            } catch (Exception e) {
                statusValueLabel.setText(e.getMessage());
                System.err.println(e.getMessage());
            }
        }
        if (!dirStructureCreated && workspace != null) {
            // Create all the corresponding subdirectories in the target
            // work subdirectories of the workspace directory.
            createDirStructure();
        }
    }//GEN-LAST:event_setCollectionItemActionPerformed

    /**
     * Creates the directory structure in subdirectories of the workspace
     * directory.
     */
    private void createDirStructure() {
        createDirStructureInternal(imagesDirectory, siftDirectory);
        createDirStructureInternal(imagesDirectory, thumbnailsDirectory);
    }

    /**
     * Creates the directory structure in subdirectories of the workspace
     * directory.
     *
     * @param imagesDir Directory containing the image files.
     * @param workDir Target subdirectory of the workspace directory.
     */
    private void createDirStructureInternal(File imagesDir, File workDir) {
        File[] children = imagesDir.listFiles();
        for (int i = 0; i < children.length; i++) {
            if (children[i].isDirectory()) {
                try {
                    FileUtil.createDirectory(new File(workDir,
                            children[i].getName()));
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
                createDirStructureInternal(children[i],
                        new File(workDir, children[i].getName()));
            }
        }
    }

    /**
     * Choose the workspace to work in.
     *
     * @param evt ActionEvent object.
     */
    private void chooseWorkspaceItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_chooseWorkspaceItemActionPerformed
        JFileChooser jfc = new JFileChooser(currentDirectory);
        jfc.setDialogTitle("Select your workspace");
        jfc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        int rVal = jfc.showOpenDialog(ImageCollectionHandler.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentDirectory = jfc.getSelectedFile();
            try {
                workspace = currentDirectory;
                workspaceValueLabel.setText(workspace.getPath());
                // Set all the corresponding files and file paths.
                clustersFile = new File(workspace, "clusteredImages.arff");
                siftDirectory = new File(workspace, "SIFT_representation");
                thumbnailsDirectory = new File(workspace, "thumbs");
                siftHistogramDirectory = new File(workspace, "SIFT_histograms");
                siftHistogramFile = new File(siftHistogramDirectory,
                        "SIFTCodebookDistributions.arff");
                siftHistogramFileNW = new File(siftHistogramDirectory,
                        "SIFTbow.arff");
                colorHistogramDirectory = new File(workspace,
                        "color_histograms");
                colorHistogramFile = new File(colorHistogramDirectory,
                        "histogramCollection.arff");
                pgmDir = new File(workspace, "temp_pgm");
                pgmFile = new File(pgmDir, "temporaryPGM.pgm");
                // Create subdirectories.
                FileUtil.createDirectory(siftDirectory);
                FileUtil.createDirectory(siftHistogramDirectory);
                FileUtil.createDirectory(colorHistogramDirectory);
                FileUtil.createDirectory(pgmDir);
                // Notify the user.
                statusValueLabel.setText("Workspace directory specified");
                if (!dirStructureCreated && imagesDirectory != null) {
                    createDirStructure();
                }
            } catch (Exception e) {
                statusValueLabel.setText(e.getMessage());
                System.err.println(e.getMessage());
            }
        }
    }//GEN-LAST:event_chooseWorkspaceItemActionPerformed

    /**
     * This method calculates the SIFT features via SiftWin.
     *
     * @param evt ActionEvent object.
     */
    private void siftItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_siftItemActionPerformed
        try {
            statusValueLabel.setText("Calculating sift features...");
            timer = new Timer(true);
            timer.scheduleAtFixedRate(new ProgressUpdater(), 1500, 1500);
            timeTimer = new javax.swing.Timer(1000, timerListener);
            timeTimer.start();
            currProgressPercentage = 0;
            // Calculations are performed in a separate thread.
            Thread t = new Thread(new SIFTHelper());
            t.start();
        } catch (Exception e) {
            statusValueLabel.setText(e.getMessage());
            System.err.println(e.getMessage());
        }
    }//GEN-LAST:event_siftItemActionPerformed

    /**
     * This method extracts the color histograms from the images.
     *
     * @param evt ActionEvent object.
     */
    private void colorsItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_colorsItemActionPerformed
        statusValueLabel.setText("Calculating color histograms...");
        timer = new Timer(true);
        timer.scheduleAtFixedRate(new ProgressUpdater(), 1500, 1500);
        timeTimer = new javax.swing.Timer(1000, timerListener);
        timeTimer.start();
        currProgressPercentage = 0;
        colorHistogramRepresentation = new DataSet();
        colorHistogramRepresentation.fAttrNames =
                new String[ColorHistogramVector.DEFAULT_NUM_BINS
                * ColorHistogramVector.DEFAULT_NUM_BINS];
        for (int i = 0; i < colorHistogramRepresentation.fAttrNames.length;
                i++) {
            colorHistogramRepresentation.fAttrNames[i] = i + "th component";
        }
        colorHistogramRepresentation.data = new ArrayList<>(numImages);
        // Calculations are performed in a separate thread.
        Thread t = new Thread(new ColorHistogramHelper());
        t.start();
    }//GEN-LAST:event_colorsItemActionPerformed

    /**
     * This method calculates the codebook for image quantization.
     *
     * @param evt ActionEvent object.
     */
    private void calculateCodeBookItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_calculateCodeBookItemActionPerformed
        JFileChooser jfc = new JFileChooser(currentDirectory);
        jfc.setDialogTitle("Select where to save codebook");
        int rVal = jfc.showSaveDialog(ImageCollectionHandler.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentDirectory = jfc.getSelectedFile().getParentFile();
            codebookFile = jfc.getSelectedFile();
            try {
                statusValueLabel.setText("Calculating codebook vectors");
                timeTimer = new javax.swing.Timer(1000, timerListener);
                timeTimer.start();
                currProgressPercentage = 0;
                cbm = new SIFTCodebookMaker(siftDirectory, true);
                timer = new Timer(true);
                timer.scheduleAtFixedRate(
                        new CodebookMakingProgressUpdater(cbm, 15), 1500, 1500);
                // Calculations are performed in a separate thread.
                Thread t = new Thread(new CodebookHelper());
                t.start();
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
        }
    }//GEN-LAST:event_calculateCodeBookItemActionPerformed

    /**
     * This method loads an existing codebook definition from the disk.
     *
     * @param evt ActionEvent object.
     */
    private void loadCodeBookItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_loadCodeBookItemActionPerformed
        JFileChooser jfc = new JFileChooser(currentDirectory);
        jfc.setDialogTitle("Select codebook file");
        jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int rVal = jfc.showOpenDialog(ImageCollectionHandler.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentDirectory = jfc.getSelectedFile().getParentFile();
            codebookFile = jfc.getSelectedFile();
            try {
                statusValueLabel.setText("Loading codebook");
                codebook = new SIFTCodeBook();
                // Perform codebook load.
                codebook.loadCodeBookFromFile(codebookFile);
                codeBookValueLabel.setText(codebookFile.getPath());
                codeBookSizeValueLabel.setText(Integer.toString(
                        codebook.getSize()));
                statusValueLabel.setText("Loaded codebook");
            } catch (Exception e) {
                statusValueLabel.setText(e.getMessage());
                System.err.println(e.getMessage());
            }
        }
    }//GEN-LAST:event_loadCodeBookItemActionPerformed

    /**
     * This method performs image quantization for all images in the data, based
     * on the currently provided codebook.
     *
     * @param evt ActionEvent object.
     */
    private void extractSIFTHistogramsItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_extractSIFTHistogramsItemActionPerformed
        try {
            Thread t = new Thread(new QuantizeAllImagesHelper());
            t.start();
        } catch (Exception e) {
            statusValueLabel.setText(e.getMessage());
            System.err.println(e.getMessage());
        }
    }//GEN-LAST:event_extractSIFTHistogramsItemActionPerformed

    /**
     * This method loads all the available data from the workspace for further
     * calculations.
     *
     * @param evt ActionEvent object.
     */
    private void loadFromWorkSpaceItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_loadFromWorkSpaceItemActionPerformed
        timeTimer = new javax.swing.Timer(1000, timerListener);
        timeTimer.start();
        Thread t = new Thread(new LoadingFromWorkspaceHelper());
        t.start();
    }//GEN-LAST:event_loadFromWorkSpaceItemActionPerformed

    /**
     * This method combines the quantized SIFT and color information into a
     * single compact image representation.
     *
     * @param evt ActionEvent object.
     */
    private void makeCombinedItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_makeCombinedItemActionPerformed
        timeTimer = new javax.swing.Timer(1000, timerListener);
        timeTimer.start();
        Thread t = new Thread(new MakeCombinedHelper());
        t.start();
    }//GEN-LAST:event_makeCombinedItemActionPerformed

    /**
     * This method saves the combined image representation to a file.
     *
     * @param evt ActionEvent object.
     */
    private void saveCombinedItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_saveCombinedItemActionPerformed
        JFileChooser jfc = new JFileChooser(currentDirectory);
        jfc.setDialogTitle("Save combined collection representations");
        jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int rVal = jfc.showSaveDialog(ImageCollectionHandler.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentDirectory = jfc.getSelectedFile().getParentFile();
            combinedFile = jfc.getSelectedFile();
            try {
                IOARFF persister = new IOARFF();
                statusValueLabel.setText("Saving representation...");
                persister.saveLabeledWithIdentifiers(combinedRepresentation,
                        combinedFile.getPath(), null);
                statusValueLabel.setText("Representation saved");
                progressBar.setValue(100);
                currProgressPercentage = 100;
            } catch (Exception e) {
                statusValueLabel.setText(e.getMessage());
                System.err.println(e.getMessage());
            }
        }
    }//GEN-LAST:event_saveCombinedItemActionPerformed

    /**
     * This method loads the combined image representation from a file.
     *
     * @param evt ActionEvent object.
     */
    private void loadCombinedItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_loadCombinedItemActionPerformed
        JFileChooser jfc = new JFileChooser(currentDirectory);
        jfc.setDialogTitle("Select combined representation file");
        jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int rVal = jfc.showOpenDialog(ImageCollectionHandler.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentDirectory = jfc.getSelectedFile().getParentFile();
            combinedFile = jfc.getSelectedFile();
            try {
                statusValueLabel.setText("Loading combined representation...");
                IOARFF persister = new IOARFF();
                combinedRepresentation = persister.load(combinedFile.getPath());
                statusValueLabel.setText("Loaded combined representation");
            } catch (Exception e) {
                statusValueLabel.setText(e.getMessage());
                System.err.println(e.getMessage());
            }
        }
    }//GEN-LAST:event_loadCombinedItemActionPerformed

    /**
     * This method saves the calculated image clusters to a file.
     *
     * @param evt ActionEvent object.
     */
    private void saveClustersItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_saveClustersItemActionPerformed
        JFileChooser jfc = new JFileChooser(currentDirectory);
        jfc.setDialogTitle("Save clusters");
        jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int rVal = jfc.showSaveDialog(ImageCollectionHandler.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentDirectory = jfc.getSelectedFile().getParentFile();
            clustersFile = jfc.getSelectedFile();
            try {
                statusValueLabel.setText("Saving representation...");
                Cluster.writeConfigurationToFile(clustersFile, clusters,
                        combinedRepresentation);
                clusterFileValueLabel.setText(clustersFile.getPath());
                statusValueLabel.setText("Representation saved");
                progressBar.setValue(100);
                currProgressPercentage = 100;
            } catch (Exception e) {
                statusValueLabel.setText(e.getMessage());
                System.err.println(e.getMessage());
            }
        }
    }//GEN-LAST:event_saveClustersItemActionPerformed

    /**
     * This method loads the representatives of the calculated image clusters
     * from a file.
     *
     * @param evt ActionEvent object.
     */
    private void loadClustersItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_loadClustersItemActionPerformed
        JFileChooser jfc = new JFileChooser(currentDirectory);
        jfc.setDialogTitle("Load clusters");
        jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int rVal = jfc.showOpenDialog(ImageCollectionHandler.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentDirectory = jfc.getSelectedFile().getParentFile();
            clustersFile = jfc.getSelectedFile();
            try {
                statusValueLabel.setText("Loading clusters...");
                clusters = Cluster.loadConfigurationFromFile(clustersFile);
                combinedRepresentation = clusters[0].getDefinitionDataset();
                Thread t = new Thread(new ClusterRepresentativesLoadHelper());
                t.start();
            } catch (Exception e) {
                statusValueLabel.setText(e.getMessage());
                statusValueLabel.setText("Load failed");
                System.err.println(e.getMessage());
            }
        }
    }//GEN-LAST:event_loadClustersItemActionPerformed

    /**
     * Read the minimal and maximal cluster numbers for clustering from the
     * input text fields.
     *
     * @param evt ActionEvent object.
     */
    private void confirmClustNumButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_confirmClustNumButtonActionPerformed
        try {
            minClust = Integer.parseInt(minClustTextField.getText().trim());
            maxClust = Integer.parseInt(maxClustTextField.getText().trim());
        } catch (Exception e) {
            minClust = -1;
            maxClust = -1;
            statusValueLabel.setText(e.getMessage());
            JOptionPane.showMessageDialog(this, e.getMessage(),
                    "Error", JOptionPane.ERROR_MESSAGE);
            System.err.println(e.getMessage());
        }
    }//GEN-LAST:event_confirmClustNumButtonActionPerformed

    /**
     * This method invokes the image clustering.
     *
     * @param evt ActionEvent object.
     */
    private void clusterImagesItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_clusterImagesItemActionPerformed
        Thread t = new Thread(new ClusterHelper(this));
        t.start();
    }//GEN-LAST:event_clusterImagesItemActionPerformed

    /**
     * Visualize the cluster representatives.
     *
     * @param evt ActionEvent object.
     */
    private void visualizeButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_visualizeButtonActionPerformed
        int clusterIndex;
        try {
            // Read which cluster to show.
            clusterIndex = Integer.parseInt(
                    inspectionNumberField.getText().trim());
            if (clusterIndex < 0 || clusterIndex >= clusters.length) {
                // Bad cluster index selection.
                throw new Exception("Must choose between: " + 0 + " and "
                        + (clusters.length - 1));
            }
            // Now visualize the cluster representatives and some basic info.
            displayedClusterNumberValueLabel.setText(
                    Integer.toString(clusterIndex));
            medoidValueLabel.setText(
                    clusterRepresentatives[clusterIndex][0].sAttr[0]);
            displayedClusterElementNumberValueLabel.setText(
                    Integer.toString(clusters[clusterIndex].size()));
            medoidPathLabel.setText(
                    clusterRepresentatives[clusterIndex][0].sAttr[0]);
            if (clusterRepresentatives[clusterIndex][1] != null) {
                secondPathLabel.setText(clusterRepresentatives[
                        clusterIndex][1].sAttr[0]);
            }
            if (clusterRepresentatives[clusterIndex][2] != null) {
                thirdPathLabel.setText(clusterRepresentatives[
                        clusterIndex][2].sAttr[0]);
            }
            imagePanel1.setImage(clusterRepresentativesImages[clusterIndex][0]);
            imagePanel2.setImage(clusterRepresentativesImages[clusterIndex][1]);
            imagePanel3.setImage(clusterRepresentativesImages[clusterIndex][2]);
        } catch (Exception e) {
            statusValueLabel.setText(e.getMessage());
            System.err.println(e.getMessage());
        }

    }//GEN-LAST:event_visualizeButtonActionPerformed

    /**
     * Re-initialize all the variables and display fields.
     *
     * @param evt ActionEvent object.
     */
    private void reinitButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_reinitButtonActionPerformed
        collectionDirValueLabel.setText("-");
        collectionSizeValueLabel.setText("-");
        workspaceValueLabel.setText("-");
        codeBookValueLabel.setText("-");
        codeBookSizeValueLabel.setText("-");
        clusterFileValueLabel.setText("-");
        executionValueLabel.setText("-");
        statusValueLabel.setText("-");
        displayedClusterNumberValueLabel.setText("-");
        medoidValueLabel.setText("-");
        displayedClusterElementNumberValueLabel.setText("-");
        medoidPathLabel.setText("---------------------------------------------"
                + "------");
        secondPathLabel.setText("----------------------------------");
        thirdPathLabel.setText("----------------------------------");
        imagePanel1.setImage(null);
        imagePanel2.setImage(null);
        imagePanel3.setImage(null);
        progressBar.setValue(0);
        currentDirectory = new File(".");
        numImages = 0;
        currProgressPercentage = 0;
        secondsElapsed = 0;
        imagesDirectory = null;
        workspace = null;
        codebookFile = null;
        if (timer != null) {
            timer.cancel();
        }
        timer = null;
        if (timeTimer != null) {
            timeTimer.stop();
        }
        timeTimer = null;
        clustersFile = null;
        siftDirectory = null;
        siftHistogramDirectory = null;
        siftHistogramFile = null;
        siftHistogramFileNW = null;
        colorHistogramDirectory = null;
        colorHistogramFile = null;
        thumbnailsDirectory = null;
        pgmDir = null;
        pgmFile = null;
        combinedFile = null;
        combinedRepresentation = null;
        colorHistogramRepresentation = null;
        codebook = null;
        cbm = null;
        siftDistributions = null;
        siftBOW = null;
        clusters = null;
        clusterRepresentatives = null;
        clusterRepresentativesImages = null;
        imageRelativePaths = null;
        dirStructureCreated = false;
        minClust = -1;
        maxClust = -1;
        validityIndex = 2;
        // Perform some clean-up.
        System.gc();
    }//GEN-LAST:event_reinitButtonActionPerformed

    /**
     * Display some basic info about the tool to the user.
     *
     * @param evt ActionEvent object.
     */
    private void aboutItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_aboutItemActionPerformed
        JOptionPane.showMessageDialog(this, "<html>Application for work with "
                + "image collections:<br>"
                + "a) Extract SIFT features for all images in a directory<br>"
                + "b) Extract color histograms<br>"
                + "c) Create new SIFT codebook or load an existing one<br>"
                + "d) Make codebook histograms for images<br>"
                + "e) Combine codebook histograms and color histograms<br>"
                + "f) Cluster with k-means pruning algorithm<br>"
                + "g) Choose the best configuration by some of the available "
                + "validity indices<br>"
                + "h) Inspect clusters by viewing some representatives for "
                + "each<br>"
                + "i) Persist clusters and/or all intermediary collections<br>"
                + "<br>"
                + "version 1.0<br>"
                + "Made by Nenad Tomasev<br>"
                + "April 2010<br>"
                + "mail suggestions to: Nenad Tomasev @gmail.com", "Info",
                JOptionPane.INFORMATION_MESSAGE);
    }//GEN-LAST:event_aboutItemActionPerformed

    /**
     * Generate image thumbnails.
     *
     * @param evt ActionEvent object.
     */
    private void makeThumbnailsItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_makeThumbnailsItemActionPerformed
        Thread t = new Thread(new ThumbnailHelper());
        t.start();
    }//GEN-LAST:event_makeThumbnailsItemActionPerformed

    /**
     * Calculate the codebook occurrence profiles.
     *
     * @param evt ActionEvent object.
     */
    private void codebookProfileMenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_codebookProfileMenuItemActionPerformed
        JFileChooser jfc = new JFileChooser(currentDirectory);
        if (codebook == null) {
            JOptionPane.showMessageDialog(this, "no codebook loaded",
                    "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }
        jfc.setDialogTitle("Save Codebook Profile: ");
        jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int rVal = jfc.showOpenDialog(ImageCollectionHandler.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentDirectory = jfc.getSelectedFile().getParentFile();
            codebookProfileFile = jfc.getSelectedFile();
            try {
                statusValueLabel.setText("Calculating codebook profile...");
                // Calculate the codebook profiles in a separate thread.
                Thread t = new Thread(new CodebookProfileHelper());
                t.start();
            } catch (Exception e) {
                statusValueLabel.setText(e.getMessage());
                statusValueLabel.setText("Calculations failed");
            }
        }
    }//GEN-LAST:event_codebookProfileMenuItemActionPerformed

    /**
     * This method makes a combined representation from the quantized SIFT and
     * color histograms.
     */
    private void makeCombinedCollection() {
        combinedRepresentation = new DataSet();
        // Set the feature names.
        combinedRepresentation.fAttrNames = new String[
                colorHistogramRepresentation.fAttrNames.length
                + siftDistributions.fAttrNames.length];
        for (int i = 0; i < colorHistogramRepresentation.fAttrNames.length;
                i++) {
            combinedRepresentation.fAttrNames[i] = "colorHistFeature_" + i;
        }
        for (int i = 0; i < siftDistributions.fAttrNames.length; i++) {
            combinedRepresentation.fAttrNames[
                    colorHistogramRepresentation.fAttrNames.length + i] =
                    "CodebookFeature_" + i;
        }
        // The relative path is kept within the nominal attributes.
        combinedRepresentation.sAttrNames = new String[1];
        combinedRepresentation.sAttrNames[0] = "relative_path";
        combinedRepresentation.data =
                new ArrayList<>(colorHistogramRepresentation.size());
        // Directories are mapped onto class indexes.
        HashMap<String, Integer> classMap =
                new HashMap<>(colorHistogramRepresentation.size() / 10);
        int classNum = 0;
        DataInstance instance;
        for (int i = 0; i < colorHistogramRepresentation.size(); i++) {
            currProgressPercentage = (int) ((float) (i * 100)
                    / (float) colorHistogramRepresentation.size());
            progressBar.setValue((int) ((float) (i * 100)
                    / (float) colorHistogramRepresentation.size()));
            statusValueLabel.setText("Combining for image: "
                    + imageRelativePaths.get(i));
            instance = new DataInstance(combinedRepresentation);
            instance.embedInDataset(combinedRepresentation);
            // Merge the feature values from the two sources.
            for (int j = 0; j < colorHistogramRepresentation.fAttrNames.length;
                    j++) {
                instance.fAttr[j] =
                        colorHistogramRepresentation.data.get(i).fAttr[j];
            }
            for (int j = 0; j < siftDistributions.fAttrNames.length; j++) {
                instance.fAttr[
                        colorHistogramRepresentation.fAttrNames.length + j] =
                        siftDistributions.data.get(i).fAttr[j];
            }
            instance.sAttr[0] = imageRelativePaths.get(i);
            // The directory is the class.
            File f = new File(imagesDirectory, imageRelativePaths.get(i));
            String className = new File(f.getParent()).getName();
            if (classMap.containsKey(className)) {
                instance.setCategory(classMap.get(className));
            } else {
                classMap.put(className, classNum++);
                instance.setCategory(classMap.get(className));
            }
            // Insert the merged data instance to the combined representation.
            combinedRepresentation.addDataInstance(instance);
        }
    }

    /**
     * @param args The command line parameters.
     */
    public static void main(String args[]) {
        java.awt.EventQueue.invokeLater(new Runnable() {
            @Override
            public void run() {
                new ImageCollectionHandler().setVisible(true);
            }
        });
    }
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JMenuItem aboutItem;
    private javax.swing.JMenuItem calculateCodeBookItem;
    private javax.swing.JMenuItem chooseWorkspaceItem;
    private javax.swing.JLabel clusterFileLabel;
    private javax.swing.JLabel clusterFileValueLabel;
    private javax.swing.JMenuItem clusterImagesItem;
    private javax.swing.JLabel clusteringIntervalLabel;
    private javax.swing.JMenu clusteringMenu;
    private javax.swing.JLabel codeBookLabel;
    private javax.swing.JMenu codeBookMenu;
    private javax.swing.JLabel codeBookSizeLabel;
    private javax.swing.JLabel codeBookSizeValueLabel;
    private javax.swing.JLabel codeBookValueLabel;
    private javax.swing.JMenuItem codebookProfileMenuItem;
    private javax.swing.JLabel collectionDirLabel;
    private javax.swing.JLabel collectionDirValueLabel;
    private javax.swing.JMenu collectionMenu;
    private javax.swing.JLabel collectionSizeLabel;
    private javax.swing.JLabel collectionSizeValueLabel;
    private javax.swing.JMenuItem colorsItem;
    private javax.swing.JMenu combinedRepresentationItem;
    private javax.swing.JButton confirmClustNumButton;
    private javax.swing.JRadioButton dbValidityRadio;
    private javax.swing.JLabel displayedClusterElementNumberLabel;
    private javax.swing.JLabel displayedClusterElementNumberValueLabel;
    private javax.swing.JLabel displayedClusterNumberLabel;
    private javax.swing.JLabel displayedClusterNumberValueLabel;
    private javax.swing.JRadioButton dunnValidityRadio;
    private javax.swing.JLabel executionLabel;
    private javax.swing.JLabel executionValueLabel;
    private javax.swing.JMenuItem extractSIFTHistogramsItem;
    private gui.images.ImagePanel imagePanel1;
    private gui.images.ImagePanel imagePanel2;
    private gui.images.ImagePanel imagePanel3;
    private javax.swing.JMenu infoMenu;
    private javax.swing.JLabel inspectLabel;
    private javax.swing.JTextField inspectionNumberField;
    private javax.swing.JMenuBar jMenuBar1;
    private javax.swing.JMenuItem loadClustersItem;
    private javax.swing.JMenuItem loadCodeBookItem;
    private javax.swing.JMenuItem loadCombinedItem;
    private javax.swing.JMenuItem loadFromWorkSpaceItem;
    private javax.swing.JMenuItem makeCombinedItem;
    private javax.swing.JMenuItem makeThumbnailsItem;
    private javax.swing.JTextField maxClustTextField;
    private javax.swing.JLabel medoidLabel;
    private javax.swing.JLabel medoidPathLabel;
    private javax.swing.JLabel medoidValueLabel;
    private javax.swing.JTextField minClustTextField;
    private javax.swing.JMenu prepareCollectionSubMenu;
    private javax.swing.JProgressBar progressBar;
    private javax.swing.JButton reinitButton;
    private javax.swing.JLabel representativeImagesLabel;
    private javax.swing.JLabel resultingNumLabel;
    private javax.swing.JLabel resultingNumValueLabel;
    private javax.swing.JMenuItem saveClustersItem;
    private javax.swing.JMenuItem saveCombinedItem;
    private javax.swing.JLabel secondPathLabel;
    private javax.swing.JMenuItem setCollectionItem;
    private javax.swing.JMenuItem siftItem;
    private javax.swing.JRadioButton silhouetteValidityRadio;
    private javax.swing.JLabel statusLabel;
    private javax.swing.JLabel statusValueLabel;
    private javax.swing.JLabel thirdPathLabel;
    private javax.swing.ButtonGroup validityButtonGroup;
    private javax.swing.JLabel validityChoiceLabel;
    private javax.swing.JButton visualizeButton;
    private javax.swing.JMenuItem visualizeItem;
    private javax.swing.JLabel workspaceLabel;
    private javax.swing.JLabel workspaceValueLabel;
    // End of variables declaration//GEN-END:variables

    /**
     * This updates the progress bar at regular intervals.
     */
    private class ProgressUpdater extends TimerTask {

        @Override
        public void run() {
            progressBar.setValue(currProgressPercentage);
        }
    }

    /**
     * This updates the clustering task progress when clustering the images.
     */
    private class ClusteringProgressUpdater extends TimerTask {

        ClusteringAlg clusterer = null;
        int priorProgress = 0;
        int max = 100;

        /**
         * Initialization.
         *
         * @param clusterer ClusteringAlg object that does the clustering.
         * @param priorProgress Progress percentage prior to invoking this
         * update mechanism.
         * @param max Integer that is the maximum progress limit, usually 100,
         * unless there is more work to be done in the task afterwards.
         */
        public ClusteringProgressUpdater(ClusteringAlg clusterer,
                int priorProgress, int max) {
            this.clusterer = clusterer;
            this.priorProgress = priorProgress;
            this.max = max;
        }

        @Override
        public void run() {
            try {
                progressBar.setValue(Math.min(99, priorProgress
                        + (int) (((float) (clusterer.getIterationIndex()))
                        / ((float) 25)) * (max - priorProgress)));
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
        }
    }

    /**
     * Updater for codebook calculation progress.
     */
    private class CodebookMakingProgressUpdater extends TimerTask {

        SIFTCodebookMaker cbm = null;
        int priorProgress;

        /**
         * Initialization.
         *
         * @param cbm SIFTCodebookMaker that does the calculations.
         * @param priorProgress Progress percentage prior to this task.
         */
        public CodebookMakingProgressUpdater(SIFTCodebookMaker cbm,
                int priorProgress) {
            this.cbm = cbm;
            this.priorProgress = priorProgress;
        }

        @Override
        public void run() {
            try {
                currProgressPercentage = priorProgress + (int) ((((float)
                        (cbm.getClusteringObject().getIterationIndex()))
                        / ((float) 25)) * (100 - priorProgress));
                progressBar.setValue(Math.min(99, currProgressPercentage));
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
        }
    }

    /**
     * Increase the value in the elapsed time display label.
     */
    private void increaseSeconds() {
        // The individual contributions are broken apart for nicer display.
        secondsElapsed++;
        int hours = (int) (secondsElapsed / 3600);
        int hoursF = (int) (hours / 10);
        int hoursS = hours % 10;
        int minutes = (int) ((secondsElapsed % 3600) / 60);
        int minutesF = (int) (minutes / 10);
        int minutesS = minutes % 10;
        int seconds = secondsElapsed % 60;
        int secondsF = (int) (seconds / 10);
        int secondsS = seconds % 10;
        executionValueLabel.setText("".concat((new Integer(hoursF)).toString()).
                concat((new Integer(hoursS)).toString()).concat(":").concat((
                new Integer(minutesF)).toString()).concat((new Integer(
                minutesS)).toString()).concat(":").concat((new Integer(
                secondsF)).toString()).concat((new Integer(secondsS)).
                toString()));
    }
    ActionListener timerListener = new ActionListener() {
        @Override
        public void actionPerformed(ActionEvent e) {
            increaseSeconds();
            try {
            } catch (Exception exc) {
                System.err.println(exc.getMessage());
            }
        }
    };

    /**
     * Cut the extension off the end of the filename in order to get a name for
     * internal use in the tool.
     *
     * @param fileName String that is the file name.
     * @return String that is the file name without the extension.
     */
    private static String nameFromFileName(String fileName) {
        int index = fileName.lastIndexOf('.');
        return fileName.substring(0, index);
    }

    /**
     * This class is used for calculating the SIFT features via SiftWin.
     */
    private class SIFTHelper implements Runnable {

        /**
         * Default constructor.
         */
        public SIFTHelper() {
        }

        @Override
        public void run() {
            File currImageFile;
            File currSIFTFile;
            for (int i = 0; i < imageRelativePaths.size(); i++) {
                currImageFile = new File(imagesDirectory,
                        imageRelativePaths.get(i));
                statusValueLabel.setText("Processing image: "
                        + currImageFile.getPath());
                currSIFTFile = new File(siftDirectory,
                        nameFromFileName(imageRelativePaths.get(i)) + ".key");
                try {
                    ConvertJPGToPGM.convertFile(currImageFile, pgmFile);
                    // Perform the extraction on the current image.
                    SiftUtil.siftFile(pgmFile, currSIFTFile, "");
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
                pgmFile.delete();
                currProgressPercentage = (int) (((float) i
                        / (float) imageRelativePaths.size()) * 100);
            }
            timer.cancel();
            timer = null;
            timeTimer.stop();
            timeTimer = null;
            currProgressPercentage = 100;
            progressBar.setValue(100);
            secondsElapsed = 0;
            statusValueLabel.setText("SIFT features generated");
        }
    }

    /**
     * This class extracts the color histograms from the images.
     */
    private class ColorHistogramHelper implements Runnable {

        /**
         * Default constructor.
         */
        public ColorHistogramHelper() {
        }

        @Override
        public void run() {
            File currImageFile;
            ColorHistogramVector colHistVector;
            for (int i = 0; i < imageRelativePaths.size(); i++) {
                if (i % 5 == 0) {
                    System.gc();
                }
                currImageFile = new File(imagesDirectory,
                        imageRelativePaths.get(i));
                statusValueLabel.setText("Processing image: "
                        + currImageFile.getPath());
                colHistVector = new ColorHistogramVector();
                try {
                    // Extract the color histogram from this image.
                    colHistVector.populateFromImage(
                            ImageIO.read(currImageFile),
                            imageRelativePaths.get(i));
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
                colorHistogramRepresentation.addDataInstance(colHistVector);
                colHistVector.embedInDataset(colorHistogramRepresentation);
                currProgressPercentage = (int) (((float) i
                        / (float) imageRelativePaths.size()) * 100);
            }
            IOARFF arff = new IOARFF();
            try {
                // Save the color histogram representation.
                arff.save(colorHistogramRepresentation,
                        colorHistogramFile.getPath(), null);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
            timer.cancel();
            timer = null;
            timeTimer.stop();
            timeTimer = null;
            currProgressPercentage = 100;
            progressBar.setValue(100);
            secondsElapsed = 0;
            statusValueLabel.setText("Histograms generated");
        }
    }

    /**
     * This class calculates the codebook for image quantization.
     */
    private class CodebookHelper implements Runnable {

        /**
         * The default constructor.
         */
        public CodebookHelper() {
        }

        @Override
        public void run() {
            try {
                statusValueLabel.setText("Loading SIFT from target "
                        + "image data...");
                cbm.getSIFTFromTargetARFFs(0.05f);
                System.out.println("Loaded...");
                currProgressPercentage = 15;
                statusValueLabel.setText("All SIFT loaded");
                statusValueLabel.setText("Finding codebook vectors...");
                // Perform feature clustering.
                cbm.clusterFeatures();
                currProgressPercentage = 99;
                // Get the codebook out.
                codebook = cbm.getCodeBookVectors();
                // Save the codebook to a file.
                codebook.writeCodeBookToFile(codebookFile);
                currProgressPercentage = 100;
                progressBar.setValue(100);
                statusValueLabel.setText("Codebooks found and saved as: "
                        + codebookFile.getName());
                codeBookValueLabel.setText(codebookFile.getPath());
                codeBookSizeValueLabel.setText(Integer.toString(
                        codebook.getSize()));
            } catch (Exception e) {
                statusValueLabel.setText(e.getMessage());
                System.err.println(e.getMessage());
            } finally {
                timeTimer.stop();
                timeTimer = null;
                timer.cancel();
                timer = null;
            }
        }
    }

    /**
     * This class calculates the codebook profiles.
     */
    private class CodebookProfileHelper implements Runnable {

        /**
         * The default constructor.
         */
        public CodebookProfileHelper() {
        }

        @Override
        public void run() {
            PrintWriter pw = null;
            try {
                statusValueLabel.setText("Calculating...");
                File currSIFTFile;
                SIFTRepresentation currSIFTRepresentation;
                QuantizedImageHistogramDataSet quantizedDSet =
                        codebook.getNewHistogramContext();
                QuantizedImageHistogram quantizedImage;
                // Directory names are mapped onto class indexes.
                HashMap<String, Integer> classMap = new HashMap<>(
                        colorHistogramRepresentation.size() / 10);
                int classNum = 0;
                // Populate the class map.
                for (int i = 0; i < colorHistogramRepresentation.size(); i++) {
                    File f = new File(imagesDirectory,
                            imageRelativePaths.get(i));
                    String className = new File(f.getParent()).getName();
                    if (!classMap.containsKey(className)) {
                        classMap.put(className, classNum++);
                    }
                }
                // Initialize the codebook profiles array.
                double[][] codebookProfiles = new double[codebook.getSize()][
                        classNum];
                for (int i = 0; i < imageRelativePaths.size(); i++) {
                    currSIFTFile = new File(siftDirectory,
                            nameFromFileName(imageRelativePaths.get(i))
                            + ".key");
                    System.out.println("Analyzing file: "
                            + currSIFTFile.getPath());
                    String className = new File(currSIFTFile.getParent()).
                            getName();
                    int currClass = classMap.get(className);
                    if (siftBOW == null || siftBOW.isEmpty()) {
                        // Load the SIFT data for the image.
                        currSIFTRepresentation =
                                SiftUtil.importFeaturesFromSift(currSIFTFile);
                        if (currSIFTRepresentation == null) {
                            currSIFTRepresentation =
                                    new SIFTRepresentation(20, 10);
                        }
                        quantizedImage = codebook.
                                getHistogramForImageRepresentation(
                                currSIFTRepresentation, quantizedDSet);
                        for (int codeIndex = 0; codeIndex < codebook.getSize();
                                codeIndex++) {
                            if (quantizedImage != null
                                    && quantizedImage.iAttr != null
                                    && quantizedImage.iAttr.length
                                    == codebook.getSize()) {
                                // Increase the total codebook occurrence count.
                                codebookProfiles[codeIndex][currClass] +=
                                        quantizedImage.iAttr[codeIndex];
                            }
                        }
                    } else {
                        DataInstance instance = siftBOW.getInstance(i);
                        for (int codeIndex = 0; codeIndex < codebook.getSize();
                                codeIndex++) {
                            // Increase the total codebook occurrence count.
                            if (instance != null && instance.iAttr != null) {
                                codebookProfiles[codeIndex][currClass] +=
                                        instance.iAttr[codeIndex];
                            }
                        }
                    }
                    progressBar.setValue((int) (((float) (i + 1) * 100)
                            / ((float) imageRelativePaths.size())));
                    if (i % 20 == 0) {
                        System.gc();
                    }
                }
                System.out.println(siftDistributions.size() + "images total");
                FileUtil.createFile(codebookProfileFile);
                pw = new PrintWriter(new FileWriter(codebookProfileFile));
                pw.println(codebook.getSize());
                for (int codeIndex = 0; codeIndex < codebook.getSize();
                        codeIndex++) {
                    pw.print(codebookProfiles[codeIndex][0]);
                    for (int cIndex = 1; cIndex < classNum; cIndex++) {
                        pw.print("," + codebookProfiles[codeIndex][cIndex]);
                    }
                    pw.println();
                }
                statusValueLabel.setText("Finished calculating...");
            } catch (Exception e) {
                System.err.println(e.getMessage());
                statusValueLabel.setText(e.getMessage());
            } finally {
                if (pw != null) {
                    pw.close();
                }
            }
        }
    }

    /**
     * This class performs batch image quantization.
     */
    private class QuantizeAllImagesHelper implements Runnable {

        /**
         * The default constructor.
         */
        public QuantizeAllImagesHelper() {
        }

        @Override
        public void run() {
            try {
                statusValueLabel.setText("Calculating...");
                File currSIFTFile;
                SIFTRepresentation currSIFTRepresentation;
                siftDistributions = codebook.getNewDistributionContext();
                siftBOW = codebook.getNewHistogramContext();
                QuantizedImageDistribution currDistr;
                QuantizedImageHistogram currHist;
                // Directory names are mapped onto class indexes.
                HashMap<String, Integer> classMap = new HashMap<>(
                        colorHistogramRepresentation.size() / 10);
                int classNum = 0;
                for (int i = 0; i < imageRelativePaths.size(); i++) {
                    // Load the data and perform the quantization.
                    currSIFTFile = new File(siftDirectory,
                            nameFromFileName(imageRelativePaths.get(i))
                            + ".key");
                    System.out.println("calculating codebook rep for file: "
                            + currSIFTFile.getPath());
                    currSIFTRepresentation = SiftUtil.importFeaturesFromSift(
                            currSIFTFile);
                    currDistr = codebook.getDistributionForImageRepresentation(
                            currSIFTRepresentation, siftDistributions);
                    currHist = codebook.getHistogramForImageRepresentation(
                            currSIFTRepresentation, siftBOW);
                    siftDistributions.addDataInstance(currDistr);
                    // Calculate the class index.
                    File f =
                            new File(imagesDirectory,
                            imageRelativePaths.get(i));
                    String className = new File(f.getParent()).getName();
                    if (classMap.containsKey(className)) {
                        currHist.setCategory(classMap.get(className));
                    } else {
                        classMap.put(className, classNum++);
                        currHist.setCategory(classMap.get(className));
                    }
                    // Insert the quantized instance into the appropriate
                    // dataset.
                    siftBOW.addDataInstance(currHist);
                    progressBar.setValue((int) (((float) (i + 1) * 100)
                            / ((float) imageRelativePaths.size())));
                    if (i % 20 == 0) {
                        System.gc();
                    }
                }
                System.out.println(siftDistributions.size() + "images total");
                IOARFF persister = new IOARFF();
                persister.save(siftDistributions, siftHistogramFile.getPath(),
                        null);
                persister = new IOARFF();
                persister.saveLabeledWithIdentifiers(siftBOW,
                        siftHistogramFileNW.getPath(), null);
                statusValueLabel.setText("Finished calculating...");
            } catch (Exception e) {
                System.err.println(e.getMessage());
                statusValueLabel.setText(e.getMessage());
            }
        }
    }

    /**
     * This class loads all the available data from the workspace directory.
     */
    private class LoadingFromWorkspaceHelper implements Runnable {

        /**
         * The default constructor.
         */
        public LoadingFromWorkspaceHelper() {
        }

        @Override
        public void run() {
            try {
                if (colorHistogramRepresentation == null) {
                    // If the histograms are not already loaded, load them.
                    statusValueLabel.setText("Loading color histograms....");
                    IOARFF arff = new IOARFF();
                    try {
                        colorHistogramRepresentation = arff.load(
                                colorHistogramFile.getPath());
                    } catch (Exception e) {
                        System.err.println(e.getMessage());
                        statusValueLabel.setText(e.getMessage());
                    }
                    statusValueLabel.setText("Loaded color histograms....");
                }
                if (siftDistributions == null) {
                    // If the SIFT distributions are not already loaded, 
                    // load them.
                    statusValueLabel.setText("Loading codebook "
                            + "distributions....");
                    IOARFF arff = new IOARFF();
                    try {
                        siftDistributions =
                                new QuantizedImageDistributionDataSet(
                                arff.load(siftHistogramFile.getPath()));
                    } catch (Exception e) {
                        System.err.println(e.getMessage());
                        statusValueLabel.setText(e.getMessage());
                    }
                    statusValueLabel.setText("Loaded codebook "
                            + "distributions....");
                }
                if (siftBOW == null || siftBOW.isEmpty()) {
                    // If the SIFT histograms are not already loaded, load them.
                    statusValueLabel.setText("Loading codebook histograms....");
                    IOARFF arff = new IOARFF();
                    try {
                        siftBOW = new QuantizedImageHistogramDataSet(
                                arff.load(siftHistogramFileNW.getPath()));
                    } catch (Exception e) {
                        System.err.println(e.getMessage());
                        statusValueLabel.setText(e.getMessage());
                    }
                    statusValueLabel.setText("Loaded codebook histograms....");
                }
                progressBar.setValue(100);
                currProgressPercentage = 100;
            } catch (Exception e) {
                System.err.println(e.getMessage());
                statusValueLabel.setText(e.getMessage());
            } finally {
                timeTimer.stop();
                timeTimer = null;
            }
        }
    }

    /**
     * This method returns the selected radio button in a button group.
     *
     * @param group ButtonGroup object.
     * @return JRadioButton that was selected.
     */
    public static JRadioButton getRadioSelection(ButtonGroup group) {
        for (Enumeration e = group.getElements(); e.hasMoreElements();) {
            JRadioButton b = (JRadioButton) e.nextElement();
            if (b.getModel() == group.getSelection()) {
                return b;
            }
        }
        return null;
    }

    /**
     * This class makes the combined SIFT and color histogram representation for
     * the loaded images.
     */
    private class MakeCombinedHelper implements Runnable {

        /**
         * The default constructor.
         */
        public MakeCombinedHelper() {
        }

        @Override
        public void run() {
            try {
                makeCombinedCollection();
            } catch (Exception e) {
                statusValueLabel.setText(e.getMessage());
            } finally {
                timeTimer.stop();
                timeTimer = null;
            }
        }
    }

    /**
     * This class performs image clustering.
     */
    private class ClusterHelper implements Runnable {

        // For updating the result fields.
        private ImageCollectionHandler handler;

        /**
         * Initialization.
         *
         * @param handler ImageCollectionHandler object reference.
         */
        public ClusterHelper(ImageCollectionHandler handler) {
            this.handler = handler;
        }

        @Override
        public void run() {
            try {
                timeTimer = new javax.swing.Timer(1000, timerListener);
                timeTimer.start();
                if (combinedRepresentation == null
                        || combinedRepresentation.isEmpty()) {
                    // If no data is provided, raise an exception.
                    JOptionPane.showMessageDialog(handler,
                            "Error, there is no data loaded to cluster",
                            "Error", JOptionPane.ERROR_MESSAGE);
                    throw new Exception(new Exception("Empty combined "
                            + "collection"));
                }
                if ((minClust < 0) || (maxClust < 0) || (minClust > maxClust)) {
                    // If the minimal and maximal cluster numbers to try have
                    // not been specified, raise an exception.
                    JOptionPane.showMessageDialog(handler,
                            "Error, incorrect min and max cluster numbers",
                            "Error", JOptionPane.ERROR_MESSAGE);
                    throw new Exception(new Exception("Incorrect interval "
                            + "input"));
                }
                // Get the proper clustering quality evaluation index.
                JRadioButton button = getRadioSelection(
                        handler.validityButtonGroup);
                if (button.equals(dunnValidityRadio)) {
                    validityIndex = DUNN;
                } else if (button.equals(dbValidityRadio)) {
                    validityIndex = DAVIES_BOULDIN;
                } else {
                    validityIndex = SILHOUETTE;
                }
                CombinedMetric cmet = new CombinedMetric(null,
                        new ColorsAndCodebooksMetric(), CombinedMetric.DEFAULT);
                Cluster[][] clusterConfiguration =
                        new Cluster[maxClust - minClust + 1][];
                FastKMeans clusterer;
                // Initiate the clustering.
                statusValueLabel.setText("Clustering...");
                for (int numClusters = minClust; numClusters <= maxClust;
                        numClusters++) {
                    // Initialize the clusterer.
                    clusterer = new FastKMeans(combinedRepresentation, cmet,
                            numClusters);
                    timer = new Timer(true);
                    // Set the progress update regimen.
                    timer.scheduleAtFixedRate(
                            new ClusteringProgressUpdater(clusterer,
                            (int) ((numClusters - minClust)
                            * ((float) 85 / (float) (maxClust - minClust + 1))),
                            (int) ((numClusters - minClust + 1) * ((float) 85
                            / (float) (maxClust - minClust + 1)))), 1500, 1500);
                    // Perform the clustering.
                    clusterer.cluster();
                    // Get the cluster configuration.
                    clusterConfiguration[numClusters - minClust] =
                            clusterer.getClusters();
                    timer.cancel();
                    timer = null;
                }
                statusValueLabel.setText("Clustering finished...");
                statusValueLabel.setText("Performing additional"
                        + "calculations...");
                // Examine cluster configuration quality for all the
                // configurations.
                OptimalConfigurationFinder selector;
                if (validityIndex == DUNN) {
                    selector = new OptimalConfigurationFinder(
                            clusterConfiguration, combinedRepresentation,
                            cmet, OptimalConfigurationFinder.DUNN_INDEX);
                } else if (validityIndex == DAVIES_BOULDIN) {
                    selector = new OptimalConfigurationFinder(
                            clusterConfiguration, combinedRepresentation, cmet,
                            OptimalConfigurationFinder.DAVIES_BOULDIN_INDEX);
                } else {
                    selector = new OptimalConfigurationFinder(
                            clusterConfiguration, combinedRepresentation, cmet,
                            OptimalConfigurationFinder.SILHOUETTE_INDEX);
                }
                // Find the optimal cluster configuration.
                clusters = selector.findBestConfiguration();
                currProgressPercentage = 90;
                progressBar.setValue(90);
                // Remove empty clusters, if any.
                clusters = ClusterConfigurationCleaner.removeEmptyClusters(
                        clusters);
                currProgressPercentage = 91;
                progressBar.setValue(91);
                resultingNumValueLabel.setText(Integer.toString(
                        clusters.length));
                clusterRepresentatives = new DataInstance[clusters.length][3];
                clusterRepresentativesImages =
                        new BufferedImage[clusters.length][3];
                float dist;
                float firstClosest;
                float secondClosest;
                // Get the list of representatives for each cluster.
                for (int clusterIndex = 0; clusterIndex < clusters.length;
                        clusterIndex++) {
                    clusterRepresentatives[clusterIndex][0] =
                            clusters[clusterIndex].getMedoid(cmet);
                    firstClosest = Float.MAX_VALUE;
                    secondClosest = Float.MAX_VALUE;
                    for (int j = 0; j < clusters[clusterIndex].size(); j++) {
                        if (!((clusters[clusterIndex].getInstance(j)).
                                equalsByContent(clusterRepresentatives[
                                clusterIndex][0]))) {
                            dist = cmet.dist(clusterRepresentatives[
                                    clusterIndex][0], clusters[clusterIndex].
                                    getInstance(j));
                            if (dist < firstClosest) {
                                clusterRepresentatives[clusterIndex][2] =
                                        clusterRepresentatives[clusterIndex][1];
                                secondClosest = firstClosest;
                                firstClosest = dist;
                                clusterRepresentatives[clusterIndex][1] =
                                        clusters[clusterIndex].getInstance(j);
                            } else if (dist < secondClosest) {
                                clusterRepresentatives[clusterIndex][2] =
                                        clusters[clusterIndex].getInstance(j);
                                secondClosest = dist;
                            }
                        }
                    }
                }
                currProgressPercentage = 95;
                progressBar.setValue(95);
                statusValueLabel.setText("Loading representative images...");
                // Load the representative images.
                File currFile;
                for (int clusterIndex = 0; clusterIndex < clusters.length;
                        clusterIndex++) {
                    for (int j = 0; j < 3; j++) {
                        if (clusterRepresentatives[clusterIndex][j] != null) {
                            currFile = new File(imagesDirectory,
                                    clusterRepresentatives[
                                    clusterIndex][j].sAttr[0]);
                            try {
                                clusterRepresentativesImages[clusterIndex][j] =
                                        ImageIO.read(currFile);
                            } catch (Exception eSecond) {
                                statusValueLabel.setText(eSecond.getMessage());
                            }
                        }
                    }
                }
                currProgressPercentage = 100;
                progressBar.setValue(100);
            } catch (Exception e) {
                statusValueLabel.setText(e.getMessage());
                System.err.println(e.getMessage());
            } finally {
                timeTimer.stop();
                timeTimer = null;
                if (timer != null) {
                    timer.cancel();
                    timer = null;
                }
            }
        }
    }

    /**
     * This class makes the image thumbnails.
     */
    private class ThumbnailHelper implements Runnable {

        /**
         * The default constructor.
         */
        public ThumbnailHelper() {
        }

        @Override
        public void run() {
            try {
                File inFile;
                BufferedImage currImage = null;
                File thumbFile = null;
                ThumbnailMaker tm = new ThumbnailMaker();
                if (imageRelativePaths != null) {
                    // Go through all the images.
                    for (int i = 0; i < imageRelativePaths.size(); i++) {
                        statusValueLabel.setText("Converting file "
                                + imageRelativePaths.get(i));
                        inFile = new File(imagesDirectory,
                                imageRelativePaths.get(i));
                        thumbFile = new File(thumbnailsDirectory,
                                imageRelativePaths.get(i));
                        try {
                            // Load the current image.
                            currImage = ImageIO.read(inFile);
                        } catch (Exception e) {
                            statusValueLabel.setText(e.getMessage());
                            System.err.println(e.getMessage());
                        }
                        try {
                            // Create and save the thumbnail.
                            ImageIO.write(tm.createThumbnail(currImage), "JPG",
                                    thumbFile);
                        } catch (Exception e) {
                            statusValueLabel.setText(e.getMessage());
                            System.err.println(e.getMessage());
                        }
                        currProgressPercentage = (int) ((float) i
                                / (float) (imageRelativePaths.size()) * 100);
                        progressBar.setValue(currProgressPercentage);
                        if (i % 10 == 0) {
                            System.gc();
                        }
                    }
                }
            } catch (Exception e) {
                statusValueLabel.setText(e.getMessage());
                System.err.println(e.getMessage());
            }
        }
    }

    /**
     * This class loads the representatives for calculated image clusters.
     */
    private class ClusterRepresentativesLoadHelper implements Runnable {

        /**
         * The default constructor.
         */
        public ClusterRepresentativesLoadHelper() {
        }

        @Override
        public void run() {
            try {
                CombinedMetric cmet = new CombinedMetric(null,
                        new Manhattan(), CombinedMetric.DEFAULT);
                resultingNumValueLabel.setText(
                        Integer.toString(clusters.length));
                clusterRepresentatives = new DataInstance[clusters.length][3];
                clusterRepresentativesImages =
                        new BufferedImage[clusters.length][3];
                float dist;
                float firstClosest;
                float secondClosest;
                currProgressPercentage = 10;
                progressBar.setValue(10);
                statusValueLabel.setText("Calculating...");
                // First calculate the representatives from the representations.
                for (int clusterIndex = 0; clusterIndex < clusters.length;
                        clusterIndex++) {
                    clusterRepresentatives[clusterIndex][0] =
                            clusters[clusterIndex].getMedoid(cmet);
                    firstClosest = Float.MAX_VALUE;
                    secondClosest = Float.MAX_VALUE;
                    for (int j = 0; j < clusters[clusterIndex].size(); j++) {
                        if (!((clusters[clusterIndex].getInstance(j)).
                                equalsByContent(clusterRepresentatives[
                                clusterIndex][0]))) {
                            dist = cmet.dist(clusterRepresentatives[
                                    clusterIndex][0], clusters[clusterIndex].
                                    getInstance(j));
                            if (dist < firstClosest) {
                                clusterRepresentatives[clusterIndex][2] =
                                        clusterRepresentatives[clusterIndex][1];
                                secondClosest = firstClosest;
                                firstClosest = dist;
                                clusterRepresentatives[clusterIndex][1] =
                                        clusters[clusterIndex].getInstance(j);
                            } else if (dist < secondClosest) {
                                clusterRepresentatives[clusterIndex][2] =
                                        clusters[clusterIndex].getInstance(j);
                                secondClosest = dist;
                            }
                        }
                    }
                }
                currProgressPercentage = 50;
                progressBar.setValue(50);
                // Load the calculated representative images from the disk.
                statusValueLabel.setText("Loading representative images...");
                File currFile;
                for (int i = 0; i < clusters.length; i++) {
                    for (int j = 0; j < 3; j++) {
                        if (clusterRepresentatives[i][j] != null) {
                            currFile = new File(imagesDirectory,
                                    clusterRepresentatives[i][j].sAttr[0]);
                            try {
                                clusterRepresentativesImages[i][j] =
                                        ImageIO.read(currFile);
                            } catch (Exception eSecond) {
                                statusValueLabel.setText(eSecond.getMessage());
                                System.err.println(eSecond.getMessage());
                            }
                        }
                    }
                }
                clusterFileValueLabel.setText(clustersFile.getPath());
                statusValueLabel.setText("Loaded clusters");
                currProgressPercentage = 100;
                progressBar.setValue(100);
            } catch (Exception e) {
                statusValueLabel.setText(e.getMessage());
                System.err.println(e.getMessage());
            } finally {
            }
        }
    }
}
