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
package gui.synthetic;

import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import distances.primary.MinkowskiMetric;
import draw.basic.ColorPalette;
import ioformat.IOARFF;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.Toolkit;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.awt.image.MemoryImageSource;
import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import javax.imageio.ImageIO;
import javax.swing.JColorChooser;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.JRadioButton;
import learning.supervised.Classifier;
import learning.supervised.methods.knn.DWHFNN;
import learning.supervised.methods.knn.HFNN;
import learning.supervised.methods.knn.HwKNN;
import learning.supervised.methods.knn.HIKNN;
import learning.supervised.methods.knn.HIKNNNonDW;
import learning.supervised.methods.knn.KNN;
import learning.supervised.methods.knn.NHBNN;
import util.BasicMathUtil;

/**
 * This class is a GUI for creating synthetic 2D data by inserting either
 * individual points or data distributions at the specified coordinates. It
 * supports multi-class data and can also generate visualizations of the kNN and
 * the reverse kNN topology, as well as classification landscapes that
 * correspond to a set of kNN methods that are implemented in this library. It
 * is possible to insert uniform noise into the data. The current implementation
 * has an upper limit on the number of possible classes to insert the instances
 * for (which can be changed), though this GUI is meant primarily for making
 * illustrative and toy examples, meaning that inserting hundreds or thousands
 * of classes would make little sense from the user perspective.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Visual2DdataGenerator extends javax.swing.JFrame {

    private JRadioButton[] classChoosers;
    // Display colors for different classes.
    private Color[] classColors;
    private int numVisibleClasses = 1;
    private File currentDirectory = new File(".");
    private File currentInFile = null;
    private File currentOutFile = null;
    // Whether to insert a point at the specified coordinates or a given
    // Gaussian distribution. Users can switch from one mode to the other.
    private boolean gaussianInsertionMode = false;

    /**
     * Creates new form Visual2DdataGenerator
     */
    public Visual2DdataGenerator() {
        initComponents();
        initialization();
    }

    /**
     * Initialize all the components.
     */
    public final void initialization() {
        // Initialize the class chooser buttons.
        classChoosers = new JRadioButton[15];
        classChoosers[0] = class0Radio;
        classChoosers[1] = class1Radio;
        classChoosers[2] = class2Radio;
        classChoosers[3] = class3Radio;
        classChoosers[4] = class4Radio;
        classChoosers[5] = class5Radio;
        classChoosers[6] = class6Radio;
        classChoosers[7] = class7Radio;
        classChoosers[8] = class8Radio;
        classChoosers[9] = class9Radio;
        classChoosers[10] = class10Radio;
        classChoosers[11] = class11Radio;
        classChoosers[12] = class12Radio;
        classChoosers[13] = class13Radio;
        classChoosers[14] = class14Radio;
        classChoosers[0].setVisible(true);
        for (int i = 1; i < classChoosers.length; i++) {
            classChoosers[i].setVisible(false);
        }
        // Initialize class colors.
        classColors = new Color[15];
        ColorPalette palette = new ColorPalette(0.4);
        classColors[0] = palette.FIREBRICK_RED;
        classColors[1] = palette.MEDIUM_SPRING_GREEN;
        classColors[2] = palette.SLATE_BLUE;
        classColors[3] = palette.DARK_WOOD;
        classColors[4] = palette.DARK_OLIVE_GREEN;
        classColors[5] = palette.ORANGE;
        classColors[6] = palette.DIM_GREY;
        classColors[7] = palette.MAROON;
        classColors[8] = palette.MEDIUM_AQUAMARINE;
        classColors[9] = palette.YELLOW_GREEN;
        classColors[10] = palette.KHAKI;
        classColors[11] = palette.PLUM;
        classColors[12] = palette.PEACH_PUFF;
        classColors[13] = palette.VIOLET;
        classColors[14] = palette.OLD_GOLD;
        for (int i = 0; i < classChoosers.length; i++) {
            classChoosers[i].setBackground(classColors[i]);
        }
        // Initialize the drawing panel for the points themselves.
        drawDSPanel.classColors = classColors;
        drawDSPanel.totalWidth = drawDSPanel.getWidth();
        drawDSPanel.totalHeight = drawDSPanel.getHeight();
        drawDSPanel.endX = drawDSPanel.getWidth();
        drawDSPanel.endY = drawDSPanel.getHeight();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        classSelectButtons = new javax.swing.ButtonGroup();
        class0Radio = new javax.swing.JRadioButton();
        class1Radio = new javax.swing.JRadioButton();
        class2Radio = new javax.swing.JRadioButton();
        class3Radio = new javax.swing.JRadioButton();
        class4Radio = new javax.swing.JRadioButton();
        class5Radio = new javax.swing.JRadioButton();
        class6Radio = new javax.swing.JRadioButton();
        class7Radio = new javax.swing.JRadioButton();
        class8Radio = new javax.swing.JRadioButton();
        class9Radio = new javax.swing.JRadioButton();
        class10Radio = new javax.swing.JRadioButton();
        class11Radio = new javax.swing.JRadioButton();
        class12Radio = new javax.swing.JRadioButton();
        class13Radio = new javax.swing.JRadioButton();
        class14Radio = new javax.swing.JRadioButton();
        addClassButton = new javax.swing.JButton();
        xNameLabel = new javax.swing.JLabel();
        yNameLabel = new javax.swing.JLabel();
        xLabel = new javax.swing.JLabel();
        yLabel = new javax.swing.JLabel();
        scalingNameLabel = new javax.swing.JLabel();
        scaleTextField = new javax.swing.JTextField();
        drawDSPanel = new gui.synthetic.DatasetDrawingPanel();
        menuBar = new javax.swing.JMenuBar();
        fileMenu = new javax.swing.JMenu();
        fileMenu.setMnemonic(KeyEvent.VK_F);
        newItem = new javax.swing.JMenuItem();
        newItem.setMnemonic(KeyEvent.VK_N);
        openItem = new javax.swing.JMenuItem();
        openItem.setMnemonic(KeyEvent.VK_O);
        saveItem = new javax.swing.JMenuItem();
        saveItem.setMnemonic(KeyEvent.VK_S);
        closeItem = new javax.swing.JMenuItem();
        closeItem.setMnemonic(KeyEvent.VK_C);
        editMenu = new javax.swing.JMenu();
        editMenu.setMnemonic(KeyEvent.VK_E);
        noiseItem = new javax.swing.JMenuItem();
        noiseItem.setMnemonic(KeyEvent.VK_N);
        mislabelItem = new javax.swing.JMenuItem();
        mislabelItem.setMnemonic(KeyEvent.VK_M);
        rotateItem = new javax.swing.JMenuItem();
        rotateItem.setMnemonic(KeyEvent.VK_R);
        undoItem = new javax.swing.JMenuItem();
        undoItem.setMnemonic(KeyEvent.VK_U);
        insertGaussianDataitem = new javax.swing.JMenuItem();
        insertGaussianDataitem.setMnemonic(KeyEvent.VK_I);
        toolsMenu = new javax.swing.JMenu();
        toolsMenu.setMnemonic(KeyEvent.VK_T);
        propertiesSubMenu = new javax.swing.JMenu();
        propertiesSubMenu.setMnemonic(KeyEvent.VK_P);
        bgColorItem = new javax.swing.JMenuItem();
        bgColorItem.setMnemonic(KeyEvent.VK_B);
        imageExportItem = new javax.swing.JMenuItem();
        imageExportItem.setMnemonic(KeyEvent.VK_E);
        knnMenu = new javax.swing.JMenu();
        knnMenu.setMnemonic(KeyEvent.VK_K);
        hubnessItem = new javax.swing.JMenu();
        hubnessLandscapeItem = new javax.swing.JMenuItem();
        hubnessLandscapeItem.setMnemonic(KeyEvent.VK_H);
        HubnessEntropyLandscapeItem = new javax.swing.JMenuItem();
        HubnessEntropyLandscapeItem.setMnemonic(KeyEvent.VK_E);
        badHubnessInterpolatedItem = new javax.swing.JMenuItem();
        badHubnessInterpolatedItem.setMnemonic(KeyEvent.VK_B);
        badHubnessKNNItem = new javax.swing.JMenuItem();
        classMapsMenu = new javax.swing.JMenu();
        knnDensityMenuItem = new javax.swing.JMenuItem();
        knnDensityMenuItem.setMnemonic(KeyEvent.VK_K);
        nhbnnProbMenuItem = new javax.swing.JMenuItem();
        nhbnnProbMenuItem.setMnemonic(KeyEvent.VK_N);
        hiknnInformationMapItem = new javax.swing.JMenuItem();
        hiknnInformationMapItem.setMnemonic(KeyEvent.VK_H);
        hiknnNonWeightedInfoItem = new javax.swing.JMenuItem();
        hwKNNDensityMenuItem = new javax.swing.JMenuItem();
        hFNNDensityMenuItem = new javax.swing.JMenuItem();
        helpMenu = new javax.swing.JMenu();
        helpMenu.setMnemonic(KeyEvent.VK_H);
        aboutItem = new javax.swing.JMenuItem();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        classSelectButtons.add(class0Radio);
        class0Radio.setText("Class 0");

        classSelectButtons.add(class1Radio);
        class1Radio.setText("Class 1");

        classSelectButtons.add(class2Radio);
        class2Radio.setText("Class 2");

        classSelectButtons.add(class3Radio);
        class3Radio.setText("Class 3");

        classSelectButtons.add(class4Radio);
        class4Radio.setText("Class 4");

        classSelectButtons.add(class5Radio);
        class5Radio.setText("Class 5");

        classSelectButtons.add(class6Radio);
        class6Radio.setText("Class 6");

        classSelectButtons.add(class7Radio);
        class7Radio.setText("Class 7");

        classSelectButtons.add(class8Radio);
        class8Radio.setText("Class 8");

        classSelectButtons.add(class9Radio);
        class9Radio.setText("Class 9");

        classSelectButtons.add(class10Radio);
        class10Radio.setText("Class 10");

        classSelectButtons.add(class11Radio);
        class11Radio.setText("Class 11");

        classSelectButtons.add(class12Radio);
        class12Radio.setText("Class 12");

        classSelectButtons.add(class13Radio);
        class13Radio.setText("Class 13");

        classSelectButtons.add(class14Radio);
        class14Radio.setText("Class 14");

        addClassButton.setText("Add Class");
        addClassButton.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                addClassButtonMouseClicked(evt);
            }
        });

        xNameLabel.setText("X:");

        yNameLabel.setText("Y:");

        xLabel.setText("0");

        yLabel.setText("0");

        scalingNameLabel.setText("setScaling factor:");

        drawDSPanel.setBackground(new java.awt.Color(255, 255, 255));
        drawDSPanel.setBorder(new javax.swing.border.MatteBorder(null));
        drawDSPanel.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                drawDSPanelMouseClicked(evt);
            }
        });
        drawDSPanel.addMouseMotionListener(new java.awt.event.MouseMotionAdapter() {
            public void mouseMoved(java.awt.event.MouseEvent evt) {
                drawDSPanelMouseMoved(evt);
            }
        });

        javax.swing.GroupLayout drawDSPanelLayout = new javax.swing.GroupLayout(drawDSPanel);
        drawDSPanel.setLayout(drawDSPanelLayout);
        drawDSPanelLayout.setHorizontalGroup(
            drawDSPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 868, Short.MAX_VALUE)
        );
        drawDSPanelLayout.setVerticalGroup(
            drawDSPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 511, Short.MAX_VALUE)
        );

        fileMenu.setText("<html><u>F</u>ile");

        newItem.setText("<html><u>N</u>ew");
        newItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                newItemActionPerformed(evt);
            }
        });
        fileMenu.add(newItem);

        openItem.setText("<html><u>O</u>pen");
        openItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                openItemActionPerformed(evt);
            }
        });
        fileMenu.add(openItem);

        saveItem.setText("<html><u>S</u>ave");
        saveItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                saveItemActionPerformed(evt);
            }
        });
        fileMenu.add(saveItem);

        closeItem.setText("<html><u>Q</u>uit");
        closeItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                closeItemActionPerformed(evt);
            }
        });
        fileMenu.add(closeItem);

        menuBar.add(fileMenu);

        editMenu.setText("<html><u>E</u>dit");

        noiseItem.setText("<html>Add Gaussian <u>N</u>oise</html>");
        noiseItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                noiseItemActionPerformed(evt);
            }
        });
        editMenu.add(noiseItem);

        mislabelItem.setText("<html>Induce <u>M</u>islabeling</html>");
        mislabelItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                mislabelItemActionPerformed(evt);
            }
        });
        editMenu.add(mislabelItem);

        rotateItem.setText("<html><u>R</u>otate</html>");
        rotateItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                rotateItemActionPerformed(evt);
            }
        });
        editMenu.add(rotateItem);

        undoItem.setText("<html><u>U</u>ndo</html>");
        undoItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                undoItemActionPerformed(evt);
            }
        });
        editMenu.add(undoItem);

        insertGaussianDataitem.setText("<html><u>I</u>nsert Gaussian Data");
        insertGaussianDataitem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                insertGaussianDataitemActionPerformed(evt);
            }
        });
        editMenu.add(insertGaussianDataitem);

        menuBar.add(editMenu);

        toolsMenu.setText("<html><u>T</u>ools");

        propertiesSubMenu.setText("<html><u>P</u>roperties");

        bgColorItem.setText("<html><u>B</u>ackground color</html>");
        bgColorItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                bgColorItemActionPerformed(evt);
            }
        });
        propertiesSubMenu.add(bgColorItem);

        toolsMenu.add(propertiesSubMenu);

        imageExportItem.setText("<html><u>E</u>xport image</html>");
        imageExportItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                imageExportItemActionPerformed(evt);
            }
        });
        toolsMenu.add(imageExportItem);

        menuBar.add(toolsMenu);

        knnMenu.setText("<html><u>K</u>NN");

        hubnessItem.setText("hubness");

        hubnessLandscapeItem.setText("<html><u>H</u>ubness landscape");
        hubnessLandscapeItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                hubnessLandscapeItemActionPerformed(evt);
            }
        });
        hubnessItem.add(hubnessLandscapeItem);

        HubnessEntropyLandscapeItem.setText("<html>Hubness <u>E</u>ntropy landscape");
        HubnessEntropyLandscapeItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                HubnessEntropyLandscapeItemActionPerformed(evt);
            }
        });
        hubnessItem.add(HubnessEntropyLandscapeItem);

        badHubnessInterpolatedItem.setText("<html><u>B</u>ad hubness interpolated");
        badHubnessInterpolatedItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                badHubnessInterpolatedItemActionPerformed(evt);
            }
        });
        hubnessItem.add(badHubnessInterpolatedItem);

        badHubnessKNNItem.setText("<html>Bad Hubness kNN");
        badHubnessKNNItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                badHubnessKNNItemActionPerformed(evt);
            }
        });
        hubnessItem.add(badHubnessKNNItem);

        knnMenu.add(hubnessItem);

        classMapsMenu.setText("classification maps");

        knnDensityMenuItem.setText("<html><u>K</u>NN probability map");
        knnDensityMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                knnDensityMenuItemActionPerformed(evt);
            }
        });
        classMapsMenu.add(knnDensityMenuItem);

        nhbnnProbMenuItem.setText("<html><u>N</u>HBNN probability map");
        nhbnnProbMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                nhbnnProbMenuItemActionPerformed(evt);
            }
        });
        classMapsMenu.add(nhbnnProbMenuItem);

        hiknnInformationMapItem.setText("<html><u>H</u>IKNN information map");
        hiknnInformationMapItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                hiknnInformationMapItemActionPerformed(evt);
            }
        });
        classMapsMenu.add(hiknnInformationMapItem);

        hiknnNonWeightedInfoItem.setText("<html>Non-Weighted HIKNN information map");
        hiknnNonWeightedInfoItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                hiknnNonWeightedInfoItemActionPerformed(evt);
            }
        });
        classMapsMenu.add(hiknnNonWeightedInfoItem);

        hwKNNDensityMenuItem.setText("hw-KNN probability map");
        hwKNNDensityMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                hwKNNDensityMenuItemActionPerformed(evt);
            }
        });
        classMapsMenu.add(hwKNNDensityMenuItem);

        hFNNDensityMenuItem.setText("h-FNN probability map");
        hFNNDensityMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                hFNNDensityMenuItemActionPerformed(evt);
            }
        });
        classMapsMenu.add(hFNNDensityMenuItem);

        knnMenu.add(classMapsMenu);

        menuBar.add(knnMenu);

        helpMenu.setText("<html><u>H</u>elp");

        aboutItem.setText("<html><u>A</u>bout");
        helpMenu.add(aboutItem);

        menuBar.add(helpMenu);

        setJMenuBar(menuBar);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(yNameLabel)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(yLabel))
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(xNameLabel)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(xLabel)
                                .addGap(231, 231, 231)
                                .addComponent(scalingNameLabel)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(scaleTextField, javax.swing.GroupLayout.PREFERRED_SIZE, 92, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addComponent(drawDSPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addGap(19, 19, 19)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(addClassButton)
                            .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                                .addComponent(class0Radio)
                                .addComponent(class1Radio)
                                .addComponent(class2Radio)
                                .addComponent(class3Radio)
                                .addComponent(class4Radio)
                                .addComponent(class5Radio)
                                .addComponent(class6Radio)
                                .addComponent(class7Radio)
                                .addComponent(class8Radio))
                            .addComponent(class9Radio)
                            .addComponent(class10Radio)
                            .addComponent(class11Radio)
                            .addComponent(class12Radio)
                            .addComponent(class13Radio)
                            .addComponent(class14Radio))))
                .addGap(20, 20, 20))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(34, 34, 34)
                        .addComponent(addClassButton)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 92, Short.MAX_VALUE)
                        .addComponent(class0Radio)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(class1Radio)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(class2Radio)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(class3Radio)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(class4Radio)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(class5Radio)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(class6Radio)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(class7Radio)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(class8Radio)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(class9Radio)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(class10Radio)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(class11Radio)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(class12Radio)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(class13Radio)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(class14Radio))
                    .addGroup(layout.createSequentialGroup()
                        .addContainerGap()
                        .addComponent(drawDSPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(xNameLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 33, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(xLabel)
                    .addComponent(scalingNameLabel)
                    .addComponent(scaleTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(yNameLabel)
                    .addComponent(yLabel))
                .addGap(44, 44, 44))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    /**
     * Inserts another class.
     *
     * @param evt MouseEvent invoked on the class insertion component.
     */
    private void addClassButtonMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_addClassButtonMouseClicked
        if (numVisibleClasses < classChoosers.length) {
            classChoosers[numVisibleClasses].setVisible(true);
            numVisibleClasses++;
        }
    }//GEN-LAST:event_addClassButtonMouseClicked

    /**
     * Determine the current class and insert a point or a set of points.
     *
     * @param evt MouseEvent on the drawing panel.
     */
    private void drawDSPanelMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_drawDSPanelMouseClicked
        Object source = evt.getSource();
        for (int i = 0; i < classChoosers.length; i++) {
            if (classChoosers[i].isSelected()) {
                drawDSPanel.currClass = i;
                break;
            }
        }
        if (source instanceof DatasetDrawingPanel) {
            if (!gaussianInsertionMode) {
                drawDSPanel.actionHistory.add(DatasetDrawingPanel.INSTANCE_ADD);
                ((DatasetDrawingPanel) source).addAndDrawPoint(evt.getX(),
                        evt.getY());
                repaint();
            } else {
                InsertGaussianDialog.showDialog(this, ((float) evt.getX()
                        / (float) drawDSPanel.getWidth()), ((float) evt.getY()
                        / (float) drawDSPanel.getHeight()));
                gaussianInsertionMode = false;
                repaint();
            }
        }
    }//GEN-LAST:event_drawDSPanelMouseClicked

    /**
     * Reset the coordinates for the current point.
     *
     * @param evt MouseEvent on the drawing panel.
     */
    private void drawDSPanelMouseMoved(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_drawDSPanelMouseMoved
        float x = BasicMathUtil.makeADecimalCutOff((float) evt.getX()
                / (float) drawDSPanel.getWidth(), 2);
        float y = BasicMathUtil.makeADecimalCutOff((float) evt.getY()
                / (float) drawDSPanel.getHeight(), 2);
        xLabel.setText((new Float(x)).toString());
        yLabel.setText((new Float(y)).toString());
    }//GEN-LAST:event_drawDSPanelMouseMoved

    /**
     * Opens an existing dataset.
     *
     * @param evt ActionEvent object.
     */
    private void openItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_openItemActionPerformed
        JFileChooser jfc = new JFileChooser(currentDirectory);
        int rVal = jfc.showOpenDialog(Visual2DdataGenerator.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentInFile = jfc.getSelectedFile();
            currentDirectory = currentInFile.getParentFile();
            IOARFF persister = new IOARFF();
            try {
                drawDSPanel.actionHistory.add(
                        DatasetDrawingPanel.DATASET_CHANGE);
                if (drawDSPanel.dset != null) {
                    drawDSPanel.allDSets.add(drawDSPanel.dset.copy());
                } else {
                    drawDSPanel.allDSets.add(null);
                }
                drawDSPanel.dset = persister.load(currentInFile.getPath());
                // Shift all the values into the positive range, hence search
                // for minX and minY.
                int numCat = drawDSPanel.dset.countCategories();
                float minX = Float.MAX_VALUE;
                float minY = Float.MAX_VALUE;
                for (int i = 0; i < drawDSPanel.dset.size(); i++) {
                    DataInstance instance = drawDSPanel.dset.data.get(i);
                    if (instance.fAttr[0] < minX) {
                        minX = instance.fAttr[0];
                    }
                    if (instance.fAttr[1] < minY) {
                        minY = instance.fAttr[1];
                    }
                }

                for (int i = 0; i < drawDSPanel.dset.size(); i++) {
                    DataInstance instance = drawDSPanel.dset.data.get(i);
                    instance.fAttr[0] += minX;
                    instance.fAttr[1] += minY;
                }

                // Scale the data into the [0,1] range.
                float maxValue = 0;
                for (int i = 0; i < drawDSPanel.dset.size(); i++) {
                    DataInstance instance = drawDSPanel.dset.data.get(i);
                    if (instance.fAttr[0] != Float.MAX_VALUE
                            && instance.fAttr[0] > maxValue) {
                        maxValue = instance.fAttr[0];
                    }
                    if (instance.fAttr[1] != Float.MAX_VALUE
                            && instance.fAttr[1] > maxValue) {
                        maxValue = instance.fAttr[1];
                    }
                }
                if (maxValue > 0) {
                    for (int i = 0; i < drawDSPanel.dset.size(); i++) {
                        DataInstance instance = drawDSPanel.dset.data.get(i);
                        instance.fAttr[0] /= maxValue;
                        instance.fAttr[1] /= maxValue;
                    }
                    scaleTextField.setText(
                            new Float(BasicMathUtil.makeADecimalCutOff(maxValue,
                            4)).toString());
                }
                while (numCat > numVisibleClasses) {
                    classChoosers[numVisibleClasses].setVisible(true);
                    numVisibleClasses++;
                }
                while (numCat < numVisibleClasses) {
                    numVisibleClasses--;
                    classChoosers[numVisibleClasses].setVisible(false);
                }
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
        }
        repaint();
    }//GEN-LAST:event_openItemActionPerformed

    /**
     * Save the generated dataset.
     *
     * @param evt ActionEvent object.
     */
    private void saveItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_saveItemActionPerformed
        JFileChooser jfc = new JFileChooser(currentDirectory);
        int rVal = jfc.showSaveDialog(Visual2DdataGenerator.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentOutFile = jfc.getSelectedFile();
            currentDirectory = currentOutFile.getParentFile();
            IOARFF persister = new IOARFF();
            try {

                DataSet dset = drawDSPanel.dset.copy();
                float scale = Float.parseFloat(scaleTextField.getText());
                // Scale the values.
                for (int i = 0; i < dset.size(); i++) {
                    dset.data.get(i).fAttr[0] *= scale;
                    dset.data.get(i).fAttr[1] *= scale;
                }
                persister.saveLabeledWithIdentifiers(dset,
                        currentOutFile.getPath(), null);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
        }
    }//GEN-LAST:event_saveItemActionPerformed

    /**
     * Change the background color.
     *
     * @param evt ActionEvent object.
     */
    private void bgColorItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_bgColorItemActionPerformed
        Color bgColor = JColorChooser.showDialog(this,
                "Select background color", Color.WHITE);
        drawDSPanel.actionHistory.add(DatasetDrawingPanel.BG_COLOR_CHANGE);
        drawDSPanel.prevColors.add(drawDSPanel.getBackground());
        drawDSPanel.setBackground(bgColor);
        repaint();
    }//GEN-LAST:event_bgColorItemActionPerformed

    /**
     * Close the GUI.
     *
     * @param evt ActionEvent object.
     */
    private void closeItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_closeItemActionPerformed
        System.exit(0);
    }//GEN-LAST:event_closeItemActionPerformed

    /**
     * Start with a new dataset.
     *
     * @param evt ActionEvent object.
     */
    private void newItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_newItemActionPerformed
        try {
            drawDSPanel.actionHistory.add(DatasetDrawingPanel.DATASET_CHANGE);
            if (drawDSPanel.dset != null) {
                drawDSPanel.allDSets.add(drawDSPanel.dset.copy());
            } else {
                drawDSPanel.allDSets.add(null);
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
        drawDSPanel.dset = null;
        for (int i = 1; i < classChoosers.length; i++) {
            classChoosers[i].setVisible(false);
        }
        numVisibleClasses = 1;
        scaleTextField.setText("");
        drawDSPanel.setBackground(Color.WHITE);
    }//GEN-LAST:event_newItemActionPerformed

    /**
     * Export the image of the drawing panel.
     *
     * @param evt ActionEvent object.
     */
    private void imageExportItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_imageExportItemActionPerformed
        JFileChooser jfc = new JFileChooser(currentDirectory);
        int rVal = jfc.showSaveDialog(Visual2DdataGenerator.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentOutFile = jfc.getSelectedFile();
            currentDirectory = currentOutFile.getParentFile();
            try {
                BufferedImage image = new BufferedImage(
                        drawDSPanel.getWidth(), drawDSPanel.getHeight(),
                        BufferedImage.TYPE_INT_RGB);
                Graphics gx = image.getGraphics();
                drawDSPanel.paint(gx);
                ImageIO.write(image, "JPG", currentOutFile);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
        }
    }//GEN-LAST:event_imageExportItemActionPerformed

    /**
     * Insert some noise into the data.
     *
     * @param evt ActionEvent object.
     */
    private void noiseItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_noiseItemActionPerformed
        String percStr = JOptionPane.showInputDialog(this,
                "Probability of noise on each attribute:",
                "Input noise probability", 1);
        float prob = Float.parseFloat(percStr);
        String stDevStr = JOptionPane.showInputDialog(this,
                "Standard deviation of Gaussian noise:",
                "Input dispersion factor", 1);
        float stDev = Float.parseFloat(stDevStr);
        try {
            drawDSPanel.actionHistory.add(DatasetDrawingPanel.DATASET_CHANGE);
            if (drawDSPanel.dset != null) {
                drawDSPanel.allDSets.add(drawDSPanel.dset.copy());
            } else {
                drawDSPanel.allDSets.add(null);
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
        drawDSPanel.dset.addGaussianNoiseToNormalizedCollection(prob, stDev);
        repaint();
    }//GEN-LAST:event_noiseItemActionPerformed

    /**
     * Introduce mislabeling into the data.
     *
     * @param evt ActionEvent object.
     */
    private void mislabelItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_mislabelItemActionPerformed
        String percStr = JOptionPane.showInputDialog(this,
                "Probability of mislabeling:",
                "Input mislabeling probability", 1);
        float prob = Float.parseFloat(percStr);
        try {
            drawDSPanel.actionHistory.add(DatasetDrawingPanel.DATASET_CHANGE);
            if (drawDSPanel.dset != null) {
                drawDSPanel.allDSets.add(drawDSPanel.dset.copy());
            } else {
                drawDSPanel.allDSets.add(null);
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
        drawDSPanel.dset.induceMislabeling(prob);
        repaint();
    }//GEN-LAST:event_mislabelItemActionPerformed

    /**
     * Perform rotation on the data.
     *
     * @param evt ActionEvent object.
     */
    private void rotateItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_rotateItemActionPerformed
        String degStr = JOptionPane.showInputDialog(this,
                "RotationAngle(degress):", "Input rotation angle", 1);
        try {
            drawDSPanel.actionHistory.add(DatasetDrawingPanel.DATASET_CHANGE);
            if (drawDSPanel.dset != null) {
                drawDSPanel.allDSets.add(drawDSPanel.dset.copy());
            } else {
                drawDSPanel.allDSets.add(null);
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
        float angleDegrees = Float.parseFloat(degStr);
        float angleDegreesRadian = (((float) (angleDegrees % 360)) / 360f)
                * 2f * (float) Math.PI;
        float angleCosine = (float) Math.cos(angleDegreesRadian);
        float angleSine = (float) Math.sin(angleDegreesRadian);
        float x, y;
        DataInstance instance;
        for (int i = 0; i < drawDSPanel.dset.size(); i++) {
            instance = drawDSPanel.dset.data.get(i);
            x = 0.5f + (instance.fAttr[0] - 0.5f) * angleCosine
                    - (instance.fAttr[1] - 0.5f) * angleSine;
            y = 0.5f + (instance.fAttr[0] - 0.5f) * angleSine
                    + (instance.fAttr[1] - 0.5f) * angleCosine;
            instance.fAttr[0] = x;
            instance.fAttr[1] = y;
        }
        float scaleFactor = Float.parseFloat(scaleTextField.getText());
        float minX = Float.MAX_VALUE;
        float minY = Float.MAX_VALUE;
        for (int i = 0; i < drawDSPanel.dset.size(); i++) {
            instance = drawDSPanel.dset.data.get(i);
            if (instance.fAttr[0] < minX) {
                minX = instance.fAttr[0];
            }
            if (instance.fAttr[1] < minY) {
                minY = instance.fAttr[1];
            }
        }

        for (int i = 0; i < drawDSPanel.dset.size(); i++) {
            instance = drawDSPanel.dset.data.get(i);
            instance.fAttr[0] += minX;
            instance.fAttr[1] += minY;
        }

        // Scale the data into the [0,1] interval.
        float maxValue = 0;
        for (int i = 0; i < drawDSPanel.dset.size(); i++) {
            instance = drawDSPanel.dset.data.get(i);
            if (instance.fAttr[0] != Float.MAX_VALUE
                    && instance.fAttr[0] > maxValue) {
                maxValue = instance.fAttr[0];
            }
            if (instance.fAttr[1] != Float.MAX_VALUE
                    && instance.fAttr[1] > maxValue) {
                maxValue = instance.fAttr[1];
            }
        }
        if (maxValue > 0) {
            for (int i = 0; i < drawDSPanel.dset.size(); i++) {
                instance = drawDSPanel.dset.data.get(i);
                instance.fAttr[0] /= maxValue;
                instance.fAttr[1] /= maxValue;
            }
            scaleTextField.setText(
                    new Float(BasicMathUtil.makeADecimalCutOff(
                    maxValue * scaleFactor, 4)).toString());
        }
        repaint();
    }//GEN-LAST:event_rotateItemActionPerformed

    /**
     * Perform undo.
     *
     * @param evt ActionEvent object.
     */
    private void undoItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_undoItemActionPerformed
        drawDSPanel.undoLast();
    }//GEN-LAST:event_undoItemActionPerformed

    /**
     * Calculate the kNN density.
     *
     * @param evt ActionEvent object.
     */
    private void knnDensityMenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_knnDensityMenuItemActionPerformed
        String kString = JOptionPane.showInputDialog("Enter k:");
        int k = Integer.parseInt(kString);
        CombinedMetric cmet = new CombinedMetric(
                null, new MinkowskiMetric(), CombinedMetric.DEFAULT);
        JFileChooser jfc = new JFileChooser(currentDirectory);
        jfc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        Random randa = new Random();
        int rVal = jfc.showOpenDialog(Visual2DdataGenerator.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentDirectory = jfc.getSelectedFile();
            int numClasses = drawDSPanel.dset.countCategories();
            System.out.println(numClasses + " " + k);
            int dsCode = randa.nextInt(10000);
            int width = drawDSPanel.getWidth();
            int height = drawDSPanel.getHeight();
            int[][] pixels = new int[numClasses][width * height];
            DataSet dset = drawDSPanel.dset;
            DataInstance instance;
            KNN classifier = new KNN(k, cmet);
            ArrayList<Integer> dIndexes = new ArrayList(dset.size());
            for (int j = 0; j < dset.size(); j++) {
                dIndexes.add(j);
            }
            classifier.setDataIndexes(dIndexes, dset);
            try {
                classifier.train();
            } catch (Exception e) {
                System.out.println("training failed: " + e.getMessage());
            }
            float[] probs = null;
            int r, g, b;
            int rgba;
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    instance = new DataInstance(dset);
                    instance.fAttr[0] = (float) x / (float) width;
                    instance.fAttr[1] = (float) y / (float) height;
                    try {
                        probs = classifier.classifyProbabilistically(instance);
                    } catch (Exception e) {
                        System.err.println("ClassificationError");
                        System.err.println(e.getMessage());
                    }
                    for (int c = 0; c < numClasses; c++) {
                        r = (int) (probs[c] * 255f);
                        g = r;
                        b = r;
                        rgba = (0xff000000 | r << 16 | g << 8 | b);
                        pixels[c][y * width + x] = rgba;
                    }
                }
            }
            for (int c = 0; c < numClasses; c++) {
                currentOutFile = new File(currentDirectory, "knnProb_"
                        + dsCode + "_" + c + ".jpg");
                BufferedImage image = new BufferedImage(width, height,
                        BufferedImage.TYPE_INT_RGB);
                Image piximg = Toolkit.getDefaultToolkit().createImage(
                        new MemoryImageSource(width, height,
                        pixels[c], 0, width));
                image.getGraphics().drawImage(piximg, 0, 0, null);
                try {
                    ImageIO.write(image, "JPG", currentOutFile);
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
            }
        }
    }//GEN-LAST:event_knnDensityMenuItemActionPerformed

    /**
     * Calculate the NHBNN class probabilities for the pixels.
     *
     * @param evt ActionEvent object.
     */
    private void nhbnnProbMenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_nhbnnProbMenuItemActionPerformed
        String kString = JOptionPane.showInputDialog("Enter k:");
        int k = Integer.parseInt(kString);
        CombinedMetric cmet = new CombinedMetric(
                null, new MinkowskiMetric(), CombinedMetric.DEFAULT);
        JFileChooser jfc = new JFileChooser(currentDirectory);
        jfc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        Random randa = new Random();
        int rVal = jfc.showOpenDialog(Visual2DdataGenerator.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentDirectory = jfc.getSelectedFile();
            int numClasses = drawDSPanel.dset.countCategories();
            System.out.println(numClasses + " " + k);
            int dsCode = randa.nextInt(10000);
            int width = drawDSPanel.getWidth();
            int height = drawDSPanel.getHeight();
            int[][] pixels = new int[numClasses][width * height];
            DataSet dset = drawDSPanel.dset;
            DataInstance instance;
            NHBNN classifier = new NHBNN(k, cmet, numClasses);
            ArrayList<Integer> dIndexes = new ArrayList(dset.size());
            for (int j = 0; j < dset.size(); j++) {
                dIndexes.add(j);
            }
            classifier.setDataIndexes(dIndexes, dset);
            try {
                classifier.train();
            } catch (Exception e) {
                System.err.println("Training failed: " + e.getMessage());
            }
            float[] probs = null;
            int r, g, b;
            int rgba;
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    instance = new DataInstance(dset);
                    instance.fAttr[0] = (float) x / (float) width;
                    instance.fAttr[1] = (float) y / (float) height;
                    try {
                        probs = classifier.classifyProbabilistically(instance);
                    } catch (Exception e) {
                        System.err.println("ClassificationError");
                        System.err.println(e.getMessage());
                    }
                    for (int c = 0; c < numClasses; c++) {
                        r = (int) (probs[c] * 255f);
                        g = r;
                        b = r;
                        rgba = (0xff000000 | r << 16 | g << 8 | b);
                        pixels[c][y * width + x] = rgba;
                    }
                }
            }
            for (int c = 0; c < numClasses; c++) {
                currentOutFile = new File(currentDirectory, "nhbnnnnProb_"
                        + dsCode + "_" + c + ".jpg");
                BufferedImage image = new BufferedImage(width, height,
                        BufferedImage.TYPE_INT_RGB);
                Image piximg = Toolkit.getDefaultToolkit().createImage(
                        new MemoryImageSource(width, height, pixels[c], 0,
                        width));
                image.getGraphics().drawImage(piximg, 0, 0, null);
                try {
                    ImageIO.write(image, "JPG", currentOutFile);
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
            }
        }
    }//GEN-LAST:event_nhbnnProbMenuItemActionPerformed

    /**
     * Calculate the HIKNN probabilities for the pixels.
     *
     * @param evt ActionEvent object.
     */
    private void hiknnInformationMapItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_hiknnInformationMapItemActionPerformed
        String kString = JOptionPane.showInputDialog("Enter k:");
        int k = Integer.parseInt(kString);
        CombinedMetric cmet = new CombinedMetric(null, new MinkowskiMetric(),
                CombinedMetric.DEFAULT);
        JFileChooser jfc = new JFileChooser(currentDirectory);
        jfc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        Random randa = new Random();
        int rVal = jfc.showOpenDialog(Visual2DdataGenerator.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentDirectory = jfc.getSelectedFile();
            int numClasses = drawDSPanel.dset.countCategories();
            System.out.println(numClasses + " " + k);
            int dsCode = randa.nextInt(10000);
            int width = drawDSPanel.getWidth();
            int height = drawDSPanel.getHeight();
            int[][] pixels = new int[numClasses][width * height];
            DataSet dset = drawDSPanel.dset;
            DataInstance instance;
            HIKNN classifier = new HIKNN(k, cmet, numClasses);
            ArrayList<Integer> dIndexes = new ArrayList(dset.size());
            for (int j = 0; j < dset.size(); j++) {
                dIndexes.add(j);
            }
            classifier.setDataIndexes(dIndexes, dset);
            try {
                classifier.train();
            } catch (Exception e) {
                System.err.println("Training failed: " + e.getMessage());
            }
            float[] probs = null;
            int r, g, b;
            int rgba;
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    instance = new DataInstance(dset);
                    instance.fAttr[0] = (float) x / (float) width;
                    instance.fAttr[1] = (float) y / (float) height;
                    try {
                        probs = classifier.classifyProbabilistically(instance);
                    } catch (Exception e) {
                        System.err.println("ClassificationError");
                        System.err.println(e.getMessage());
                    }
                    for (int c = 0; c < numClasses; c++) {
                        r = (int) (probs[c] * 255f);
                        g = r;
                        b = r;
                        rgba = (0xff000000 | r << 16 | g << 8 | b);
                        pixels[c][y * width + x] = rgba;
                    }
                }
            }
            for (int c = 0; c < numClasses; c++) {
                currentOutFile = new File(currentDirectory, "hiknnInfo_"
                        + dsCode + "_" + c + ".jpg");
                BufferedImage image = new BufferedImage(width, height,
                        BufferedImage.TYPE_INT_RGB);
                Image piximg = Toolkit.getDefaultToolkit().createImage(
                        new MemoryImageSource(width, height, pixels[c], 0,
                        width));
                image.getGraphics().drawImage(piximg, 0, 0, null);
                try {
                    ImageIO.write(image, "JPG", currentOutFile);
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
            }
        }
    }//GEN-LAST:event_hiknnInformationMapItemActionPerformed

    /**
     * Calculate the HIKNN class probabilities for each pixel, without the
     * distance weighting.
     *
     * @param evt ActionEvent object.
     */
    private void hiknnNonWeightedInfoItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_hiknnNonWeightedInfoItemActionPerformed
        String kString = JOptionPane.showInputDialog("Enter k:");
        int k = Integer.parseInt(kString);
        CombinedMetric cmet = new CombinedMetric(null, new MinkowskiMetric(),
                CombinedMetric.DEFAULT);
        JFileChooser jfc = new JFileChooser(currentDirectory);
        jfc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        Random randa = new Random();
        int rVal = jfc.showOpenDialog(Visual2DdataGenerator.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentDirectory = jfc.getSelectedFile();
            int numClasses = drawDSPanel.dset.countCategories();
            System.out.println(numClasses + " " + k);
            int dsCode = randa.nextInt(10000);
            int width = drawDSPanel.getWidth();
            int height = drawDSPanel.getHeight();
            int[][] pixels = new int[numClasses][width * height];
            DataSet dset = drawDSPanel.dset;
            DataInstance instance;
            HIKNNNonDW classifier = new HIKNNNonDW(k, cmet, numClasses);
            ArrayList<Integer> dIndexes = new ArrayList(dset.size());
            for (int j = 0; j < dset.size(); j++) {
                dIndexes.add(j);
            }
            classifier.setDataIndexes(dIndexes, dset);
            try {
                classifier.train();
            } catch (Exception e) {
                System.err.println("Training failed: " + e.getMessage());
            }
            float[] probs = null;
            int r, g, b;
            int rgba;
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    instance = new DataInstance(dset);
                    instance.fAttr[0] = (float) x / (float) width;
                    instance.fAttr[1] = (float) y / (float) height;
                    try {
                        probs = classifier.classifyProbabilistically(instance);
                    } catch (Exception e) {
                        System.err.println("ClassificationError");
                        System.err.println(e.getMessage());
                    }
                    for (int c = 0; c < numClasses; c++) {
                        r = (int) (probs[c] * 255f);
                        g = r;
                        b = r;
                        rgba = (0xff000000 | r << 16 | g << 8 | b);
                        pixels[c][y * width + x] = rgba;
                    }
                }
            }
            for (int c = 0; c < numClasses; c++) {
                currentOutFile = new File(currentDirectory,
                        "hiknnNonWeightedInfo_" + dsCode + "_" + c + ".jpg");
                BufferedImage image = new BufferedImage(width, height,
                        BufferedImage.TYPE_INT_RGB);
                Image piximg = Toolkit.getDefaultToolkit().createImage(
                        new MemoryImageSource(width, height, pixels[c], 0,
                        width));
                image.getGraphics().drawImage(piximg, 0, 0, null);
                try {
                    ImageIO.write(image, "JPG", currentOutFile);
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
            }
        }
    }//GEN-LAST:event_hiknnNonWeightedInfoItemActionPerformed

    /**
     * Calculate class probabilities according to hw-kNN for each pixel.
     *
     * @param evt ActionEvent object.
     */
    private void hwKNNDensityMenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_hwKNNDensityMenuItemActionPerformed
        String kString = JOptionPane.showInputDialog("Enter k:");
        int k = Integer.parseInt(kString);
        CombinedMetric cmet = new CombinedMetric(null, new MinkowskiMetric(),
                CombinedMetric.DEFAULT);
        JFileChooser jfc = new JFileChooser(currentDirectory);
        jfc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        Random randa = new Random();
        int rVal = jfc.showOpenDialog(Visual2DdataGenerator.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentDirectory = jfc.getSelectedFile();
            int numClasses = drawDSPanel.dset.countCategories();
            System.out.println(numClasses + " " + k);
            int dsCode = randa.nextInt(10000);
            int width = drawDSPanel.getWidth();
            int height = drawDSPanel.getHeight();
            int[][] pixels = new int[numClasses][width * height];
            DataSet dset = drawDSPanel.dset;
            DataInstance instance;
            HwKNN classifier = new HwKNN(numClasses, cmet, k);
            ArrayList<Integer> dIndexes = new ArrayList(dset.size());
            for (int j = 0; j < dset.size(); j++) {
                dIndexes.add(j);
            }
            classifier.setDataIndexes(dIndexes, dset);
            try {
                classifier.train();
            } catch (Exception e) {
                System.out.println("Training failed: " + e.getMessage());
            }
            float[] probs = null;
            int r, g, b;
            int rgba;
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    instance = new DataInstance(dset);
                    instance.fAttr[0] = (float) x / (float) width;
                    instance.fAttr[1] = (float) y / (float) height;
                    try {
                        probs = classifier.classifyProbabilistically(instance);
                    } catch (Exception e) {
                        System.out.println("ClassificationError");
                        System.out.println(e.getMessage());
                    }
                    for (int c = 0; c < numClasses; c++) {
                        r = (int) (probs[c] * 255f);
                        g = r;
                        b = r;
                        rgba = (0xff000000 | r << 16 | g << 8 | b);
                        pixels[c][y * width + x] = rgba;
                    }
                }
            }
            for (int c = 0; c < numClasses; c++) {
                currentOutFile = new File(currentDirectory, "hwKNN_"
                        + dsCode + "_" + c + ".jpg");
                BufferedImage image = new BufferedImage(width, height,
                        BufferedImage.TYPE_INT_RGB);
                Image piximg = Toolkit.getDefaultToolkit().createImage(
                        new MemoryImageSource(width, height, pixels[c], 0,
                        width));
                image.getGraphics().drawImage(piximg, 0, 0, null);
                try {
                    ImageIO.write(image, "JPG", currentOutFile);
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
            }
        }
    }//GEN-LAST:event_hwKNNDensityMenuItemActionPerformed

    /**
     * Calculates the class probabilities according to hFNN for all pixels.
     *
     * @param evt ActionEvent object.
     */
    private void hFNNDensityMenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_hFNNDensityMenuItemActionPerformed
        String kString = JOptionPane.showInputDialog("Enter k:");
        int k = Integer.parseInt(kString);
        int dWeighted = JOptionPane.showConfirmDialog(this,
                "Use distance weighting?", "Distance weighting",
                JOptionPane.YES_NO_OPTION);
        boolean useDistanceWeighting;
        if (dWeighted == JOptionPane.YES_OPTION) {
            useDistanceWeighting = true;
        } else {
            useDistanceWeighting = false;
        }
        CombinedMetric cmet = new CombinedMetric(null, new MinkowskiMetric(),
                CombinedMetric.DEFAULT);
        JFileChooser jfc = new JFileChooser(currentDirectory);
        jfc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        Random randa = new Random();
        int rVal = jfc.showOpenDialog(Visual2DdataGenerator.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentDirectory = jfc.getSelectedFile();
            int numClasses = drawDSPanel.dset.countCategories();
            System.out.println(numClasses + " " + k);
            int dsCode = randa.nextInt(10000);
            int width = drawDSPanel.getWidth();
            int height = drawDSPanel.getHeight();
            int[][] pixels = new int[numClasses][width * height];
            DataSet dset = drawDSPanel.dset;
            DataInstance instance;
            Classifier classifier = null;
            if (useDistanceWeighting) {
                classifier = new DWHFNN(k, cmet, numClasses);
            } else {
                classifier = new HFNN(k, cmet, numClasses);
            }
            ArrayList<Integer> dIndexes = new ArrayList(dset.size());
            for (int j = 0; j < dset.size(); j++) {
                dIndexes.add(j);
            }
            classifier.setDataIndexes(dIndexes, dset);
            try {
                classifier.train();
            } catch (Exception e) {
                System.out.println("training failed: " + e.getMessage());
            }
            float[] probs = null;
            int r, g, b;
            int rgba;
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    instance = new DataInstance(dset);
                    instance.fAttr[0] = (float) x / (float) width;
                    instance.fAttr[1] = (float) y / (float) height;
                    try {
                        probs = classifier.classifyProbabilistically(instance);
                    } catch (Exception e) {
                        System.err.println("classificationError");
                        System.err.println(e.getMessage());
                    }
                    for (int c = 0; c < numClasses; c++) {
                        r = (int) (probs[c] * 255f);
                        g = r;
                        b = r;
                        rgba = (0xff000000 | r << 16 | g << 8 | b);
                        pixels[c][y * width + x] = rgba;
                    }
                }
            }
            for (int c = 0; c < numClasses; c++) {
                if (useDistanceWeighting) {
                    currentOutFile = new File(currentDirectory, "hdwFNN_"
                            + dsCode + "_" + c + ".jpg");
                } else {
                    currentOutFile = new File(currentDirectory, "hFNN_"
                            + dsCode + "_" + c + ".jpg");
                }
                BufferedImage image = new BufferedImage(width, height,
                        BufferedImage.TYPE_INT_RGB);
                Image piximg = Toolkit.getDefaultToolkit().createImage(
                        new MemoryImageSource(width, height, pixels[c], 0,
                        width));
                image.getGraphics().drawImage(piximg, 0, 0, null);
                try {
                    ImageIO.write(image, "JPG", currentOutFile);
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
            }
        }
    }//GEN-LAST:event_hFNNDensityMenuItemActionPerformed

    /**
     * Calculate the hubness landscape - how often a point would occur in the
     * kNN sets of other points, for each pixel.
     *
     * @param evt
     */
    private void hubnessLandscapeItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_hubnessLandscapeItemActionPerformed
        String kString = JOptionPane.showInputDialog("Enter k:");
        int k = Integer.parseInt(kString);
        CombinedMetric cmet = new CombinedMetric(null, new MinkowskiMetric(),
                CombinedMetric.DEFAULT);
        JFileChooser jfc = new JFileChooser(currentDirectory);
        jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int rVal = jfc.showOpenDialog(Visual2DdataGenerator.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentOutFile = jfc.getSelectedFile();
            int width = drawDSPanel.getWidth();
            int height = drawDSPanel.getHeight();
            float[] hubness = new float[width * height];
            int[] pixels = new int[width * height];
            DataSet dset = drawDSPanel.dset;
            DataInstance tempInstance;
            NeighborSetFinder nsf = new NeighborSetFinder(dset, cmet);
            try {
                nsf.calculateDistances();
                nsf.calculateNeighborSets(k);
            } catch (Exception e) {
                System.err.println("Neighbor set error: " + e.getMessage());
            }
            int r, g, b;
            int rgba;
            float[][] kDistances = nsf.getKDistances();
            float maxHubness = 0;
            float currDist = 0;
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    tempInstance = new DataInstance(dset);
                    tempInstance.fAttr[0] = (float) x / (float) width;
                    tempInstance.fAttr[1] = (float) y / (float) height;
                    for (int index = 0; index < dset.size(); index++) {
                        try {
                            currDist = cmet.dist(tempInstance,
                                    dset.data.get(index));
                        } catch (Exception e) {
                            System.err.println("Distance error for point: "
                                    + x + " " + y + " : " + e.getMessage());
                        }
                        // We compare the current distance to the k-distance
                        // of the kNN set.
                        if (currDist < kDistances[index][k - 1]) {
                            hubness[y * width + x]++;
                        }
                    }
                    if (hubness[y * width + x] > maxHubness) {
                        maxHubness = hubness[y * width + x];
                    }
                }
            }
            // Normalization.
            for (int hi = 0; hi < hubness.length; hi++) {
                hubness[hi] /= maxHubness;
            }
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    r = (int) (hubness[y * width + x] * 255f);
                    g = r;
                    b = r;
                    rgba = (0xff000000 | r << 16 | g << 8 | b);
                    pixels[y * width + x] = rgba;
                }
            }
            BufferedImage image = new BufferedImage(width, height,
                    BufferedImage.TYPE_INT_RGB);
            Image piximg = Toolkit.getDefaultToolkit().createImage(
                    new MemoryImageSource(width, height, pixels, 0, width));
            image.getGraphics().drawImage(piximg, 0, 0, null);
            try {
                ImageIO.write(image, "JPG", currentOutFile);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
        }
    }//GEN-LAST:event_hubnessLandscapeItemActionPerformed

    /**
     * Calculate the reverse neighbor set entropies for all the pixels.
     *
     * @param evt ActionEvent object.
     */
    private void HubnessEntropyLandscapeItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_HubnessEntropyLandscapeItemActionPerformed
        String kString = JOptionPane.showInputDialog("Enter k:");
        int k = Integer.parseInt(kString);
        CombinedMetric cmet = new CombinedMetric(null, new MinkowskiMetric(),
                CombinedMetric.DEFAULT);
        int numClasses = drawDSPanel.dset.countCategories();
        JFileChooser jfc = new JFileChooser(currentDirectory);
        jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int rVal = jfc.showOpenDialog(Visual2DdataGenerator.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentOutFile = jfc.getSelectedFile();
            int width = drawDSPanel.getWidth();
            int height = drawDSPanel.getHeight();
            float[][] classConditionalHubness = new float[width * height][numClasses];
            float[] hubness = new float[width * height];
            float[] hubnessEntropies = new float[width * height];
            int[] pixels = new int[width * height];
            DataSet dset = drawDSPanel.dset;
            DataInstance instance;
            // Calculate the kNN sets.
            NeighborSetFinder nsf = new NeighborSetFinder(dset, cmet);
            try {
                nsf.calculateDistances();
                nsf.calculateNeighborSets(k);
            } catch (Exception e) {
                System.err.println("Neighbor set error: " + e.getMessage());
            }
            int r, g, b;
            int rgba;
            float[][] kDistances = nsf.getKDistances();
            float maxHubnessEntropy = 0;
            float currDist = 0;
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    instance = new DataInstance(dset);
                    instance.fAttr[0] = (float) x / (float) width;
                    instance.fAttr[1] = (float) y / (float) height;
                    for (int index = 0; index < dset.size(); index++) {
                        try {
                            currDist = cmet.dist(instance,
                                    dset.data.get(index));
                        } catch (Exception e) {
                            System.err.println("Distance error for point: "
                                    + x + " " + y + " : " + e.getMessage());
                        }
                        if (currDist < kDistances[index][k - 1]) {
                            hubness[y * width + x]++;
                            classConditionalHubness[y * width + x][
                                    dset.data.get(index).getCategory()]++;
                        }
                    }
                    // Calculate the entropies.
                    if (hubness[y * width + x] > 0) {
                        for (int c = 0; c < numClasses; c++) {
                            classConditionalHubness[y * width + x][c] /=
                                    hubness[y * width + x];
                            if (classConditionalHubness[y * width + x][c] > 0) {
                                hubnessEntropies[y * width + x] -=
                                        BasicMathUtil.log2(
                                        classConditionalHubness[
                                        y * width + x][c]);
                            }
                        }
                    } else {
                        hubnessEntropies[y * width + x] = 0;
                    }
                    if (hubnessEntropies[y * width + x] > maxHubnessEntropy) {
                        maxHubnessEntropy = hubness[y * width + x];
                    }
                }
            }
            // Normalize.
            for (int hi = 0; hi < hubness.length; hi++) {
                hubnessEntropies[hi] /= maxHubnessEntropy;
            }
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    r = (int) (hubnessEntropies[y * width + x] * 255f);
                    g = r;
                    b = r;
                    rgba = (0xff000000 | r << 16 | g << 8 | b);
                    pixels[y * width + x] = rgba;
                }
            }
            BufferedImage image = new BufferedImage(width, height,
                    BufferedImage.TYPE_INT_RGB);
            Image piximg = Toolkit.getDefaultToolkit().createImage(
                    new MemoryImageSource(width, height, pixels, 0, width));
            image.getGraphics().drawImage(piximg, 0, 0, null);
            try {
                ImageIO.write(image, "JPG", currentOutFile);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
        }
    }//GEN-LAST:event_HubnessEntropyLandscapeItemActionPerformed

    /**
     * Calculates the average regional bad hubness of each pixel point after by
     * integrating over all the points while taking their distances into
     * account.
     *
     * @param evt ActionEvent object.
     */
    private void badHubnessInterpolatedItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_badHubnessInterpolatedItemActionPerformed
        String kString = JOptionPane.showInputDialog("Enter k:");
        int k = Integer.parseInt(kString);
        CombinedMetric cmet = new CombinedMetric(null, new MinkowskiMetric(),
                CombinedMetric.DEFAULT);
        JFileChooser jfc = new JFileChooser(currentDirectory);
        jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int rVal = jfc.showOpenDialog(Visual2DdataGenerator.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentOutFile = jfc.getSelectedFile();
            int width = drawDSPanel.getWidth();
            int height = drawDSPanel.getHeight();
            float[] goodHubness = new float[width * height];
            int[] pixels = new int[width * height];
            DataSet dset = drawDSPanel.dset;
            DataInstance instance;
            // Calculate the kNN sets.
            NeighborSetFinder nsf = new NeighborSetFinder(dset, cmet);
            try {
                nsf.calculateDistances();
                nsf.calculateNeighborSets(k);
            } catch (Exception e) {
                System.err.println("Neighbor set error: " + e.getMessage());
            }
            float[] instanceWeights = nsf.getHWKNNWeightingScheme();
            int r, g, b;
            int rgba;
            float maxGoodHubness = -Float.MAX_VALUE;
            float minGoodHubness = Float.MAX_VALUE;
            float currDist = 0;
            float[] instDists = new float[dset.size()];
            float totalDist;
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    totalDist = 0;
                    instance = new DataInstance(dset);
                    instance.fAttr[0] = (float) x / (float) width;
                    instance.fAttr[1] = (float) y / (float) height;
                    for (int index = 0; index < dset.size(); index++) {
                        try {
                            currDist = cmet.dist(instance,
                                    dset.data.get(index));
                        } catch (Exception e) {
                            System.err.println("Distance error for point: "
                                    + x + " " + y + " : " + e.getMessage());
                        }
                        instDists[index] = currDist;
                        totalDist += currDist;
                    }
                    for (int index = 0; index < dset.size(); index++) {
                        instDists[index] /= totalDist;
                        goodHubness[y * width + x] +=
                                (instDists[index] * instanceWeights[index]);
                    }

                }
            }
            // Perform good hubness normalization.
            for (int i = 0; i < goodHubness.length; i++) {
                if (goodHubness[i] > maxGoodHubness) {
                    maxGoodHubness = goodHubness[i];
                }
                if (goodHubness[i] < minGoodHubness) {
                    minGoodHubness = goodHubness[i];
                }
            }
            for (int i = 0; i < goodHubness.length; i++) {
                goodHubness[i] = (goodHubness[i] - minGoodHubness)
                        / (maxGoodHubness - minGoodHubness);
            }
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    r = (int) ((1 - goodHubness[y * width + x]) * 255f);
                    g = (int) ((goodHubness[y * width + x]) * 255f);
                    b = 0;
                    rgba = (0xff000000 | r << 16 | g << 8 | b);
                    pixels[y * width + x] = rgba;
                }
            }
            BufferedImage image = new BufferedImage(width, height,
                    BufferedImage.TYPE_INT_RGB);
            Image piximg = Toolkit.getDefaultToolkit().createImage(
                    new MemoryImageSource(width, height, pixels, 0, width));
            image.getGraphics().drawImage(piximg, 0, 0, null);
            try {
                ImageIO.write(image, "JPG", currentOutFile);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
        }
    }//GEN-LAST:event_badHubnessInterpolatedItemActionPerformed

    /**
     * Calculates the bad hubness scores for all the pixels.
     *
     * @param evt ActionEvent object.
     */
    private void badHubnessKNNItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_badHubnessKNNItemActionPerformed
        String kString = JOptionPane.showInputDialog("Enter k:");
        int k = Integer.parseInt(kString);
        CombinedMetric cmet = new CombinedMetric(null, new MinkowskiMetric(),
                CombinedMetric.DEFAULT);
        JFileChooser jfc = new JFileChooser(currentDirectory);
        jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int rVal = jfc.showOpenDialog(Visual2DdataGenerator.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentOutFile = jfc.getSelectedFile();
            int width = drawDSPanel.getWidth();
            int height = drawDSPanel.getHeight();

            float[] goodHubness = new float[width * height];
            int[] pixels = new int[width * height];
            DataSet dset = drawDSPanel.dset;
            DataInstance instance;
            // Calculate the kNN sets.
            NeighborSetFinder nsf = new NeighborSetFinder(dset, cmet);
            try {
                nsf.calculateDistances();
                nsf.calculateNeighborSets(k);
            } catch (Exception e) {
                System.err.println("Neighbor set error: " + e.getMessage());
            }
            float[] instanceWeights = nsf.getHWKNNWeightingScheme();
            int r, g, b;
            int rgba;
            float maxGoodHubness = -Float.MAX_VALUE;
            float minGoodHubness = Float.MAX_VALUE;
            float currDist = 0;
            float[] instDists = new float[k];
            float totalDist;
            int[] kNeighborIndexes = null;
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    totalDist = 0;
                    instance = new DataInstance(dset);
                    instance.fAttr[0] = (float) x / (float) width;
                    instance.fAttr[1] = (float) y / (float) height;
                    try {
                        kNeighborIndexes = NeighborSetFinder.
                                getIndexesOfNeighbors(dset, instance, k, cmet);
                    } catch (Exception e) {
                        System.err.println("KNN error for point: " + x + " "
                                + y + " : " + e.getMessage());
                    }
                    for (int index = 0; index < k; index++) {
                        try {
                            currDist = cmet.dist(instance, dset.data.get(
                                    kNeighborIndexes[index]));
                        } catch (Exception e) {
                            System.out.println("Distance error for point: "
                                    + x + " " + y + " : " + e.getMessage());
                        }
                        instDists[index] = currDist;
                        totalDist += currDist;
                    }
                    for (int index = 0; index < k; index++) {
                        instDists[index] /= totalDist;
                        goodHubness[y * width + x] += (instDists[index]
                                * instanceWeights[kNeighborIndexes[index]]);
                    }
                }
            }
            // Perform good hubness normalization.
            for (int i = 0; i < goodHubness.length; i++) {
                if (goodHubness[i] > maxGoodHubness) {
                    maxGoodHubness = goodHubness[i];
                }
                if (goodHubness[i] < minGoodHubness) {
                    minGoodHubness = goodHubness[i];
                }
            }
            for (int i = 0; i < goodHubness.length; i++) {
                goodHubness[i] = (goodHubness[i] - minGoodHubness)
                        / (maxGoodHubness - minGoodHubness);
            }

            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    r = (int) ((1 - goodHubness[y * width + x]) * 255f);
                    g = (int) ((goodHubness[y * width + x]) * 255f);
                    b = 0;
                    rgba = (0xff000000 | r << 16 | g << 8 | b);
                    pixels[y * width + x] = rgba;
                }
            }
            BufferedImage image = new BufferedImage(width, height,
                    BufferedImage.TYPE_INT_RGB);
            Image piximg = Toolkit.getDefaultToolkit().createImage(
                    new MemoryImageSource(width, height, pixels, 0, width));
            image.getGraphics().drawImage(piximg, 0, 0, null);
            try {
                ImageIO.write(image, "JPG", currentOutFile);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
        }
    }//GEN-LAST:event_badHubnessKNNItemActionPerformed

    /**
     * Sets the Gaussian data insertion mode.
     *
     * @param evt ActionEvent object.
     */
    private void insertGaussianDataitemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_insertGaussianDataitemActionPerformed
        gaussianInsertionMode = true;
    }//GEN-LAST:event_insertGaussianDataitemActionPerformed

    /**
     * @param args The command line arguments.
     */
    public static void main(String args[]) {
        java.awt.EventQueue.invokeLater(new Runnable() {
            @Override
            public void run() {
                new Visual2DdataGenerator().setVisible(true);
            }
        });
    }
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JMenuItem HubnessEntropyLandscapeItem;
    private javax.swing.JMenuItem aboutItem;
    private javax.swing.JButton addClassButton;
    private javax.swing.JMenuItem badHubnessInterpolatedItem;
    private javax.swing.JMenuItem badHubnessKNNItem;
    private javax.swing.JMenuItem bgColorItem;
    private javax.swing.JRadioButton class0Radio;
    private javax.swing.JRadioButton class10Radio;
    private javax.swing.JRadioButton class11Radio;
    private javax.swing.JRadioButton class12Radio;
    private javax.swing.JRadioButton class13Radio;
    private javax.swing.JRadioButton class14Radio;
    private javax.swing.JRadioButton class1Radio;
    private javax.swing.JRadioButton class2Radio;
    private javax.swing.JRadioButton class3Radio;
    private javax.swing.JRadioButton class4Radio;
    private javax.swing.JRadioButton class5Radio;
    private javax.swing.JRadioButton class6Radio;
    private javax.swing.JRadioButton class7Radio;
    private javax.swing.JRadioButton class8Radio;
    private javax.swing.JRadioButton class9Radio;
    private javax.swing.JMenu classMapsMenu;
    private javax.swing.ButtonGroup classSelectButtons;
    private javax.swing.JMenuItem closeItem;
    private gui.synthetic.DatasetDrawingPanel drawDSPanel;
    private javax.swing.JMenu editMenu;
    private javax.swing.JMenu fileMenu;
    private javax.swing.JMenuItem hFNNDensityMenuItem;
    private javax.swing.JMenu helpMenu;
    private javax.swing.JMenuItem hiknnInformationMapItem;
    private javax.swing.JMenuItem hiknnNonWeightedInfoItem;
    private javax.swing.JMenu hubnessItem;
    private javax.swing.JMenuItem hubnessLandscapeItem;
    private javax.swing.JMenuItem hwKNNDensityMenuItem;
    private javax.swing.JMenuItem imageExportItem;
    private javax.swing.JMenuItem insertGaussianDataitem;
    private javax.swing.JMenuItem knnDensityMenuItem;
    private javax.swing.JMenu knnMenu;
    private javax.swing.JMenuBar menuBar;
    private javax.swing.JMenuItem mislabelItem;
    private javax.swing.JMenuItem newItem;
    private javax.swing.JMenuItem nhbnnProbMenuItem;
    private javax.swing.JMenuItem noiseItem;
    private javax.swing.JMenuItem openItem;
    private javax.swing.JMenu propertiesSubMenu;
    private javax.swing.JMenuItem rotateItem;
    private javax.swing.JMenuItem saveItem;
    private javax.swing.JTextField scaleTextField;
    private javax.swing.JLabel scalingNameLabel;
    private javax.swing.JMenu toolsMenu;
    private javax.swing.JMenuItem undoItem;
    private javax.swing.JLabel xLabel;
    private javax.swing.JLabel xNameLabel;
    private javax.swing.JLabel yLabel;
    private javax.swing.JLabel yNameLabel;
    // End of variables declaration//GEN-END:variables

    /**
     * Insert Gaussian data points.
     *
     * @param x Float that is the current X coordinate.
     * @param y Float that is the current Y coordinate.
     * @param xSigma Float that is the standard deviation in the X direction.
     * @param ySigma Float that is the standard deviation in the Y direction.
     * @param rotAngle Float that is the rotation angle.
     * @param numInstances Integer that is the number of instances.
     */
    public void insertGaussian(float x, float y, float xSigma, float ySigma,
            float rotAngle, int numInstances) {
        Random randa = new Random();
        if (numInstances <= 0) {
            return;
        }
        if (xSigma <= 0) {
            return;
        }
        if (ySigma <= 0) {
            return;
        }
        // Convert to radians.
        float rotRadian = rotAngle / 180f * (float) Math.PI;
        float X;
        float Y;
        float Xtemp;
        float Ytemp;
        float cos = (float) (Math.cos(rotRadian));
        float sin = (float) (Math.sin(rotRadian));
        for (int i = 0; i < numInstances; i++) {
            Xtemp = (float) randa.nextGaussian() * xSigma;
            Ytemp = (float) randa.nextGaussian() * ySigma;
            // Now rotate.
            X = Math.max(Math.min(cos * Xtemp - sin * Ytemp + x, 1), 0);
            Y = Math.max(Math.min(sin * Xtemp + cos * Ytemp + y, 1), 0);
            // Insert the point into the panel at the specified coordinates.
            drawDSPanel.addAndDrawNormalizedPoint(X, Y);
        }
    }
}
