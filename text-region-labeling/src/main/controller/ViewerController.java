package main.controller;

import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.input.*;
import javafx.scene.layout.AnchorPane;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.util.Pair;
import main.App;
import main.utils.PointProcessor;
import main.utils.XmlFileWriter;

import javax.annotation.PostConstruct;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class ViewerController {

    @FXML
    private AnchorPane rootPane;
    @FXML
    private ImageView imageView;
    @FXML
    private TextField outputField;
    @FXML
    private Button writeBtn;
    @FXML
    private CheckBox previewCheckBox;
    @FXML
    private Button regionBtn;
    @FXML
    private Button boundaryLineBtn;
    @FXML
    private Button baseLineBtn;
    @FXML
    private Button textLineBtn;
    @FXML
    private CheckBox axisCheckBox;

    private LinkedList<Pair<Integer, Integer>> trace = new LinkedList<>();
    private LinkedList<Pair<Integer, Integer>> redoStack = new LinkedList<>();
    private XmlFileWriter xmlFileWriter;
    private Image originalImage, preDrawnImage;
    private String originalImageFileName;
    private BrowserController browserController;
    private StringBuilder groundTruthTxt = null;
    private boolean haveDrawnRegion = false;
    private boolean canBeWritten = false;
    private List<int[]> tempBoundary = null;
    private List<int[]> tempBaseLine = null;
    private double maxX, maxY;


    @PostConstruct
    public void initialize() {
        rootPane.setOnKeyPressed(this::handleKeyboardShortcuts);
        imageView.setOnMouseClicked(this::handleCheckpointAction);
        outputField.setOnKeyPressed(this::handleKeyboardShortcuts);

        if (!previewCheckBox.isSelected()) {
            previewCheckBox.fire();
        }
        imageView.setOnMouseMoved(this::handleMouseMoveAction);
        writeBtn.setOnAction(event -> handleWriteAction());
        regionBtn.setOnMouseClicked(this::handleAddTextRegionAction);
        boundaryLineBtn.setOnMouseClicked(this::handleAddBoundaryLineAction);
        baseLineBtn.setOnMouseClicked(this::handleAddBaseLineAction);
        textLineBtn.setOnMouseClicked(this::handleAddTextLineAction);
        axisCheckBox.setOnMouseReleased(this::handleTriggerAxisCheckBox);
        imageView.setOnScroll(this::handleZoomImageAction);
    }

    private void handleKeyboardShortcuts(KeyEvent event) {
        if (event.isControlDown() && event.getCode() == KeyCode.Z) {
            if (!event.isShiftDown()) {
                undo();
            } else {
                redo();
            }
        } else if (event.getCode() == KeyCode.ESCAPE && previewCheckBox.isSelected()) {
            previewCheckBox.fire();
        } else if (event.getCode() == KeyCode.I) {
            scale(2d);
        } else if (event.getCode() == KeyCode.O) {
            scale(0.5d);
        } else {
            return;
        }
        event.consume();
    }

    private void scale(Double scaleFactor) {
        imageView.setScaleX(imageView.getScaleX() * scaleFactor);
        imageView.setScaleY(imageView.getScaleY() * scaleFactor);
    }

    private void handleWriteAction() {
        //TO-DO: check some conditions to assure have enough data
        if (!canBeWritten) {
            showErrorDialog("You have to add the text line first!");
            return;
        }

        File outputFolder = browserController.getOutputFolder();
        try {
            String fileName = this.originalImageFileName;
            fileName = fileName.substring(0, fileName.lastIndexOf('.')) + ".xml";
            File outputFile = new File(outputFolder.getPath() + '/' + fileName);
            xmlFileWriter.writeXmlFile(outputFile);
            startOverDrawImage();
        } catch (NullPointerException e) {
            handleFileNotFound(outputFolder);
            return;
        }

        showInfoDialog("Your XML file is written successfully!");
        Stage stage = (Stage) writeBtn.getScene().getWindow();
        stage.close();
    }

    private void handleFileNotFound(File outputFile) {
        Alert alert = new Alert(Alert.AlertType.ERROR,
                "The folder is not found. Update the path in the main window and retry.",
                ButtonType.OK);
        alert.setHeaderText("File not found!");
        alert.initModality(Modality.WINDOW_MODAL);
        alert.showAndWait().ifPresent(buttonType -> browserController.openXmlFileBrowser());
    }

    private void handleCheckpointAction(MouseEvent event) {
        int x = (int) event.getX();
        int y = (int) event.getY();
        trace.push(new Pair<>(x, y));
        redoStack.clear();
        updateOutput();
    }

    private void handleMouseMoveAction(MouseEvent event) {
        if (previewCheckBox.isSelected()) {
            int x = (int) event.getX();
            int y = (int) event.getY();
            Pair<Integer, Integer> current = new Pair<>(x, y);
            trace.push(current);
            updateOutput();
            trace.poll();
        }
        if (axisCheckBox.isSelected()) {
            handleDrawAxis(event);
        }
    }

    private void handleAddTextRegionAction(MouseEvent event) {
        if (event.getButton() != MouseButton.PRIMARY) return;
        if (trace.isEmpty()) {
            showErrorDialog("You have to draw some points on the image!");
            return;
        }

        String temp = PointProcessor.listToString(PointProcessor.linkedlistToList(trace));
        xmlFileWriter.addTextRegion(temp);
        haveDrawnRegion = true;
        startOverDrawImage();
    }

    private void handleAddBoundaryLineAction(MouseEvent event) {
        if (event.getButton() != MouseButton.PRIMARY) return;
        if (trace.isEmpty()) {
            showErrorDialog("You have to draw some points on the image!");
            return;
        }

        tempBoundary = PointProcessor.linkedlistToList(trace);
        startOverDrawImage();
    }

    private void handleAddBaseLineAction (MouseEvent event) {
        if (event.getButton() != MouseButton.PRIMARY) return;
        if (trace.isEmpty()) {
            showErrorDialog("You have to draw some points on the image!");
            return;
        }

        tempBaseLine = PointProcessor.linkedlistToList(trace);
        startOverDrawImage();
    }

    private void handleAddTextLineAction (MouseEvent event) {
        if (event.getButton() != MouseButton.PRIMARY) return;

        StringBuilder ret = new StringBuilder();
        openTextDialog(ret);

        if (!haveDrawnRegion) {
            showErrorDialog("You have not drawn the region yet. Please check again!");
            return;
        }
        if (tempBoundary == null) {
            showErrorDialog("You have not drawn boundary of the line yet. Please check again!");
            return;
        }
        if (tempBaseLine == null) {
            showErrorDialog("You have not drawn base of the line yet. Please check again!");
            return;
        }
        if (ret.length() == 0) {
            return;
        }

        String groundTruthLine = ret.toString();
        int idx = groundTruthTxt.indexOf(groundTruthLine);
        groundTruthTxt.delete(idx, idx + ret.length());
        groundTruthTxt = new StringBuilder(groundTruthTxt.toString().trim());

        PointProcessor.harmonyBoundaryAndBaseLine(tempBoundary, tempBaseLine);

        xmlFileWriter.addTextLine(PointProcessor.listToString(tempBoundary), PointProcessor.listToString(tempBaseLine), groundTruthLine);
        canBeWritten = true;
    }

    private void openTextDialog(StringBuilder output) {
        try {
            FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("../view/GroundTruthTextSelectionDialog.fxml"));
            Parent loader = fxmlLoader.load();
            GroundTruthTextSelectionController dialogController = fxmlLoader.getController();
            dialogController.setProperty(output, groundTruthTxt);

            Stage viewerStage = new Stage();
            viewerStage.setTitle("Text Selection Dialog");
            viewerStage.setMinHeight(350);
            viewerStage.setMinWidth(480);
            viewerStage.setScene(new Scene(loader, 550, 430));
            viewerStage.showAndWait();

        } catch (Exception e) {
            App.showExceptionAlert(e);
        }
    }

    private void undo() {
        Pair<Integer, Integer> current = trace.poll();
        if (current != null) {
            redoStack.push(current);
            updateOutput();
        }
    }

    private void redo() {
        Pair<Integer, Integer> current = redoStack.poll();
        if (current != null) {
            trace.push(current);
            updateOutput();
        }
    }

    private void drawPolygon() {
        BufferedImage bufferedImage = SwingFXUtils.fromFXImage(originalImage, null);
        Graphics2D graphics = (Graphics2D) bufferedImage.getGraphics();
        graphics.setStroke(new BasicStroke(2));
        graphics.setColor(Color.RED);
        if (!trace.isEmpty()) {
            Pair<Integer, Integer> start = trace.getLast();
            Iterator<Pair<Integer, Integer>> iterator = trace.descendingIterator();
            Pair<Integer, Integer> current = iterator.next();
            while (iterator.hasNext()) {
                Pair<Integer, Integer> next = iterator.next();
                graphics.drawLine(current.getKey(), current.getValue(), next.getKey(), next.getValue());
                current = next;
            }
            graphics.drawLine(current.getKey(), current.getValue(), start.getKey(), start.getValue());
        }
        WritableImage writableImage = SwingFXUtils.toFXImage(bufferedImage, null);
        imageView.setImage(writableImage);
    }

    private void updateOutput() {
        drawPolygon();
        preDrawnImage = imageView.getImage();
        StringBuilder text = new StringBuilder();
        Iterator<Pair<Integer, Integer>> iterator = trace.descendingIterator();
        while (iterator.hasNext()) {
            Pair<Integer, Integer> coordinate = iterator.next();
            if (coordinate != null) {
                text.append(coordinate.getKey()).append(",").append(coordinate.getValue()).append(" ");
            }
        }
        outputField.setText(text.toString().trim());
        //writeBtn.setDisable(trace.isEmpty());
    }

    private void startOverDrawImage() {
        trace.clear();
        redoStack.clear();
        imageView.setImage(originalImage); // flush the polygon
        updateOutput();
    }

    public void setOriginalImage(Image image, String imageFileName) {
        originalImage = image;
        preDrawnImage = originalImage;
        originalImageFileName = imageFileName;
        imageView.setImage(image);
        maxX = imageView.getImage().getWidth();
        maxY = imageView.getImage().getHeight();
    }

    public void setBrowserController(BrowserController browserController) {
        this.browserController = browserController;
    }

    public void createXmlFileWriter(String fileName, int imageWidth, int imageHeight) {
        xmlFileWriter = new XmlFileWriter(fileName, imageWidth, imageHeight);
    }

    public void setGroundTruthTxt(String txt) {
        groundTruthTxt = new StringBuilder(txt);
    }

    private void showErrorDialog(String content) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle("Error Dialog");
        alert.setContentText(content);
        alert.showAndWait();
    }

    private void showInfoDialog(String content) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("Information Dialog");
        alert.setContentText(content);
        alert.showAndWait();
    }

    private void handleDrawAxis(MouseEvent event) {
        Image img = previewCheckBox.isSelected() ? imageView.getImage() : preDrawnImage;
        BufferedImage bufferedImage = SwingFXUtils.fromFXImage(img, null);
        Graphics2D graphics = (Graphics2D) bufferedImage.getGraphics();
        graphics.setStroke(new BasicStroke(2));
        graphics.setColor(Color.BLUE);

        int x = (int) event.getX();
        int y = (int) event.getY();

        graphics.drawLine(x, y, 0, y);
        graphics.drawLine(x, y, (int) maxX, y);
        graphics.drawLine(x, y, x, 0);
        graphics.drawLine(x, y, x, (int) maxY);

        WritableImage writableImage = SwingFXUtils.toFXImage(bufferedImage, null);
        imageView.setImage(writableImage);
    }

    private void handleTriggerAxisCheckBox(MouseEvent event) {
        if (event.getButton() != MouseButton.PRIMARY) return;

        if (!axisCheckBox.isSelected()) {
            imageView.setImage(preDrawnImage);
        }
    }

    private void handleZoomImageAction(ScrollEvent event) {
        double zoomFactor = 1.05;
        double deltaY = event.getDeltaY();
        if (deltaY < 0) {
            zoomFactor = 2.0 - zoomFactor;
        }
        scale(zoomFactor);
    }
}
