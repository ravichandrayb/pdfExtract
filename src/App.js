import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [pdfImage, setPdfImage] = useState(null);
  const [regions, setRegions] = useState([]);
  const [currentRegion, setCurrentRegion] = useState({ name: '', x1: 0, y1: 0, x2: 0, y2: 0 });
  const [isDrawing, setIsDrawing] = useState(false);
  const [fileName, setFileName] = useState("");
  const [file, setFile] = useState({});
  const [extractedText, setExtractedText] = useState({});
  const canvasRef = useRef(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const handleFileUpload = async (event) => {

    
    const file = event.target.files[0];
    setFileName(file.name)
    setFile(file)
    //console.log(file,"file")
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/upload-pdf', formData);
      setPdfImage(response.data.image_path);
    } catch (error) {
      console.error('Error uploading PDF:', error);
    }
  };
  
  const handleExtractText = async () => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/extract-text/', formData);
      setExtractedText(response.data);
      alert('Text extracted successfully');
    } catch (error) {
      console.error('Error extracting text:', error);
    }
  };

  const handleCanvasMouseDown = (event) => {
    const { offsetX, offsetY } = event.nativeEvent;
    setCurrentRegion({ ...currentRegion, x1: offsetX, y1: offsetY });
    setIsDrawing(true);
  };

  const handleCanvasMouseMove = (event) => {
    if (!isDrawing) return;
    const { offsetX, offsetY } = event.nativeEvent;
    setCurrentRegion({ ...currentRegion, x2: offsetX, y2: offsetY });
  };

  const handleCanvasMouseUp = () => {
    setIsDrawing(false);
    if (currentRegion.name) {
      setRegions([...regions, currentRegion]);
      setCurrentRegion({ name: '', x1: 0, y1: 0, x2: 0, y2: 0 });
    }
  };

  const handleNextROI = () => {
    const name = prompt('Enter a name for the next region of interest:');
    if (name) {
      setCurrentRegion({ ...currentRegion, name });
    }
  };

  const handleFinish = async () => {
    try {
      await axios.post('http://localhost:8000/save-roi', regions,fileName);
      alert('ROI coordinates have been saved successfully');
    } catch (error) {
      console.error('Error saving ROI coordinates:', error);
    }
  };

  useEffect(() => {
    if (pdfImage) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const img = new Image();
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        regions.forEach(region => {
          ctx.strokeStyle = 'red';
          ctx.lineWidth = 2;
          ctx.strokeRect(region.x1, region.y1, region.x2 - region.x1, region.y2 - region.y1);
        });
        if (isDrawing) {
          ctx.strokeStyle = 'black';
          ctx.lineWidth = 2;
          ctx.strokeRect(currentRegion.x1, currentRegion.y1, currentRegion.x2 - currentRegion.x1, currentRegion.y2 - currentRegion.y1);
        }
      };
      img.src = `http://localhost:8000/static/${pdfImage}`;
    }
  }, [pdfImage, regions, currentRegion, isDrawing]);

  return (
    <div className="App">
      <input type="file" onChange={handleFileUpload} accept=".pdf" />
      <button onClick={handleNextROI}>Next ROI</button>
      <button onClick={handleFinish}>Finish</button>
      <button onClick={handleExtractText}>Extract Text</button>
      <canvas
        ref={canvasRef}
        onMouseDown={handleCanvasMouseDown}
        onMouseMove={handleCanvasMouseMove}
        onMouseUp={handleCanvasMouseUp}
      />
      <table>
        <thead>
          <tr>
            <th>ROI</th>
            <th>x1</th>
            <th>y1</th>
            <th>x2</th>
            <th>y2</th>
          </tr>
        </thead>
        <tbody>
          {regions.map((region, index) => (
            <tr key={index}>
              <td>{region.name}</td>
              <td>{region.x1}</td>
              <td>{region.y1}</td>
              <td>{region.x2}</td>
              <td>{region.y2}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div>
        <h3>Extracted Text:</h3>
        <pre>{JSON.stringify(extractedText, null, 2)}</pre>
      </div>
    </div>
  );
}

export default App;