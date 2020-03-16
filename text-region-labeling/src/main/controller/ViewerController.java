package main.controller;

import javafx.embed.swing.SwingFXUtils;
import javafx.event.Event;
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
import javafx.scene.input.KeyCode;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.AnchorPane;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.util.Pair;
import main.App;
import main.controller.BrowserController;
import main.utils.XmlFileWriter;

import javax.annotation.PostConstruct;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.Iterator;
import java.util.LinkedList;

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

    private LinkedList<Pair<Integer, Integer>> trace = new LinkedList<>();
    private LinkedList<Pair<Integer, Integer>> redoStack = new LinkedList<>();
    private XmlFileWriter xmlFileWriter;
    private Image originalImage;
    private String originalImageFileName;
    private BrowserController browserController;
    private StringBuilder groundTruthTxt = null;
    private String tempBoundary = null;
    private String tempBaseLine = null;


    @PostConstruct
    public void initialize() {
        imageView.setOnMouseClicked(this::handleCheckpointAction);
        rootPane.setOnKeyPressed(event -> {
            if (event.isControlDown() && event.getCode() == KeyCode.Z) {
                if (!event.isShiftDown()) {
                    undo();
                } else {
                    redo();
                }
            } else if (event.getCode() == KeyCode.ESCAPE && previewCheckBox.isSelected()) {
                previewCheckBox.fire();
            }
        });
        previewCheckBox.setOnAction(event -> {
            if (previewCheckBox.isSelected()) {
                enablePreview();
            } else {
                disablePreview();
            }
        });
        if (!previewCheckBox.isSelected()) {
            previewCheckBox.fire();
        }
        writeBtn.setOnAction(event -> handleWriteAction());
        regionBtn.setOnMouseClicked(this::handleAddTextRegionAction);
        boundaryLineBtn.setOnMouseClicked(this::handleAddBoundaryLineAction);
        baseLineBtn.setOnMouseClicked(this::handleAddBaseLineAction);
        textLineBtn.setOnMouseClicked(this::handleAddTextLineAction);
    }

    private void handleWriteAction() {
        if (!outputField.getText().isEmpty()) {
            File outputFile = browserController.getOutputFile();
            try (FileWriter fileWriter = new FileWriter(outputFile, true)) {
                PrintWriter printWriter = new PrintWriter(fileWriter);
                printWriter.println(originalImageFileName + " " + outputField.getText());
                trace.clear();
                redoStack.clear();
                originalImage = imageView.getImage(); // flush the polygon
                updateOutput();
            } catch (NullPointerException | FileNotFoundException e) {
                handleFileNotFound(outputFile);
            } catch (IOException e) {
                App.showExceptionAlert(e);
            }
        }
    }

    private void handleFileNotFound(File outputFile) {
        Alert alert = new Alert(Alert.AlertType.ERROR,
                String.format("The file: %s%nis not found. Update the path in the main window and retry.", outputFile),
                ButtonType.OK);
        alert.setHeaderText("File not found!");
        alert.initModality(Modality.WINDOW_MODAL);
        alert.showAndWait().ifPresent(buttonType -> browserController.openTextFileBrowser());
    }

    private void handleCheckpointAction(MouseEvent event) {
        int x = (int) event.getX();
        int y = (int) event.getY();
        trace.push(new Pair<>(x, y));
        redoStack.clear();
        updateOutput();
    }

    private void handlePreviewAction(MouseEvent event) {
        int x = (int) event.getX();
        int y = (int) event.getY();
        Pair<Integer, Integer> current = new Pair<>(x, y);
        trace.push(current);
        updateOutput();
        trace.poll();
    }

    private void handleAddTextRegionAction(MouseEvent event) {
        if (!event.isPrimaryButtonDown()) return;
        if (outputField.getText().isEmpty()) return;

        xmlFileWriter.addTextRegion(outputField.getText());
    }

    private void handleAddBoundaryLineAction(MouseEvent event) {
        if (!event.isPrimaryButtonDown()) return;
        if (outputField.getText().isEmpty()) return;

        tempBoundary = outputField.getText();
    }

    private void handleAddBaseLineAction (MouseEvent event) {
        if (!event.isPrimaryButtonDown()) return;
        if (outputField.getText().isEmpty()) return;

        tempBaseLine = outputField.getText();
    }

    private void handleAddTextLineAction (MouseEvent event) {
        //if (!event.isPrimaryButtonDown()) return;
        //if (tempBoundary == null || tempBaseLine == null) return;

        openTextDialog();
    }

    private void openTextDialog() {
        try {
            FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("../view/GroundTruthTextSelectionDialog.fxml"));
            Parent loader = fxmlLoader.load();
            GroundTruthTextSelectionController dialogController = fxmlLoader.getController();
            StringBuilder outputTxt = new StringBuilder();
            dialogController.setProperty(outputTxt, groundTruthTxt);

            Stage viewerStage = new Stage();
            viewerStage.setTitle("Text Selection Dialog");
            viewerStage.setMinHeight(350);
            viewerStage.setMinWidth(480);
            viewerStage.setScene(new Scene(loader, 550, 430));
            viewerStage.show();

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
        StringBuilder text = new StringBuilder();
        Iterator<Pair<Integer, Integer>> iterator = trace.descendingIterator();
        while (iterator.hasNext()) {
            Pair<Integer, Integer> coordinate = iterator.next();
            if (coordinate != null) {
                text.append(coordinate.getKey()).append(",").append(coordinate.getValue()).append(" ");
            }
        }
        outputField.setText(text.toString().trim());
        writeBtn.setDisable(trace.isEmpty());
    }

    public void setOriginalImage(Image image, String imageFileName) {
        originalImage = image;
        originalImageFileName = imageFileName;
        imageView.setImage(image);
    }

    private void disablePreview() {
        updateOutput();
        imageView.setOnMouseMoved(Event::consume);
    }

    private void enablePreview() {
        imageView.setOnMouseMoved(this::handlePreviewAction);
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
}
