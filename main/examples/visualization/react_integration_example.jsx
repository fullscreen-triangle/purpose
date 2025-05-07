// Example React component showing how to integrate sprint model visualizations
// This file demonstrates how the RAG system can be connected to D3 visualizations
// in a React-based frontend application

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';

// This would be your actual React component for the RAG UI
const SprintModelRAG = () => {
  const [query, setQuery] = useState('');
  const [modelResult, setModelResult] = useState(null);
  const [visualizationCode, setVisualizationCode] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('visualization');
  const [error, setError] = useState(null);
  
  // Function to submit a query to the RAG backend
  const submitQuery = async () => {
    if (!query.trim()) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      // Call your RAG backend API
      const response = await axios.post('/api/query', { query });
      
      // Update with the model result
      setModelResult(response.data.result);
      
      // Request a visualization for this model
      if (response.data.result) {
        const vizResponse = await axios.post('/api/visualize', { 
          modelResult: response.data.result 
        });
        
        setVisualizationCode(vizResponse.data.visualization);
      }
    } catch (err) {
      console.error('Error querying the RAG system:', err);
      setError('An error occurred while processing your query. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };
  
  // This component dynamically renders the generated visualization
  const DynamicVisualization = ({ code }) => {
    const [Component, setComponent] = useState(null);
    
    useEffect(() => {
      if (!code) return;
      
      try {
        // Clean the code to ensure it's a valid React component
        let cleanCode = code;
        
        // Convert the string code to an actual component
        // WARNING: This is using eval which has security implications
        // In a production environment, use a safer approach
        const ComponentFunction = new Function('React', 'd3', 'props', 
          `${cleanCode}; return React.createElement(SprintVisualization, props);`
        );
        
        // Import d3 dynamically
        import('d3').then(d3 => {
          setComponent(() => props => ComponentFunction(React, d3, props));
        });
      } catch (err) {
        console.error('Error rendering visualization component:', err);
      }
    }, [code]);
    
    if (!Component) return <div>Loading visualization...</div>;
    
    return <Component />;
  };
  
  return (
    <div className="sprint-model-rag">
      <h1>Sprint Model RAG System</h1>
      
      <div className="query-section">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about sprint modeling or optimization... (e.g., 'Create a mathematical model for sprint velocity as a function of time, accounting for wind resistance of 2m/s')"
          rows={4}
        />
        <button 
          onClick={submitQuery} 
          disabled={isLoading}
        >
          {isLoading ? 'Processing...' : 'Submit Query'}
        </button>
      </div>
      
      {error && <div className="error-message">{error}</div>}
      
      {modelResult && (
        <div className="results-section">
          <h2>Model Result</h2>
          <div className="model-result">
            <SyntaxHighlighter language="markdown" style={tomorrow}>
              {modelResult}
            </SyntaxHighlighter>
          </div>
          
          <div className="visualization-container">
            <div className="tabs">
              <button 
                className={activeTab === 'visualization' ? 'active' : ''} 
                onClick={() => setActiveTab('visualization')}
              >
                Visualization
              </button>
              <button 
                className={activeTab === 'code' ? 'active' : ''} 
                onClick={() => setActiveTab('code')}
              >
                Code
              </button>
            </div>
            
            <div className="tab-content">
              {activeTab === 'visualization' ? (
                visualizationCode ? (
                  <div className="visualization-view">
                    <DynamicVisualization code={visualizationCode} />
                  </div>
                ) : (
                  <div className="no-visualization">
                    No visualization available for this model.
                  </div>
                )
              ) : (
                <div className="code-view">
                  <SyntaxHighlighter language="jsx" style={tomorrow}>
                    {visualizationCode || '// No visualization code available'}
                  </SyntaxHighlighter>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SprintModelRAG;

// Safer alternative to dynamic component evaluation
// This component can be used to render pre-approved visualizations
const SafeVisualizationRenderer = ({ type, data }) => {
  // Map of approved visualization components
  const visualizations = {
    velocity: React.lazy(() => import('./visualizations/VelocityChart')),
    stride: React.lazy(() => import('./visualizations/StrideChart')),
    ground_reaction: React.lazy(() => import('./visualizations/GroundReactionChart')),
    energy: React.lazy(() => import('./visualizations/EnergyChart')),
    // Add other visualization types here
  };
  
  const VisualizationComponent = visualizations[type];
  
  if (!VisualizationComponent) {
    return <div>Visualization type not supported: {type}</div>;
  }
  
  return (
    <React.Suspense fallback={<div>Loading visualization...</div>}>
      <VisualizationComponent data={data} />
    </React.Suspense>
  );
};

// Example API endpoints implementation (for reference)
/*
// Backend Express.js route that would handle the RAG query
app.post('/api/query', async (req, res) => {
  try {
    const { query } = req.body;
    
    // Call your Python RAG system
    const { spawn } = require('child_process');
    const python = spawn('python', ['rag_query.py', query]);
    
    let result = '';
    python.stdout.on('data', (data) => {
      result += data.toString();
    });
    
    python.on('close', (code) => {
      if (code !== 0) {
        return res.status(500).json({ error: 'Error executing RAG query' });
      }
      res.json({ result });
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Backend route to generate visualizations
app.post('/api/visualize', async (req, res) => {
  try {
    const { modelResult } = req.body;
    
    // Call your Python visualization generator
    const { spawn } = require('child_process');
    const python = spawn('python', ['generate_visualization.py']);
    
    // Send the model result to the Python script
    python.stdin.write(modelResult);
    python.stdin.end();
    
    let visualization = '';
    python.stdout.on('data', (data) => {
      visualization += data.toString();
    });
    
    python.on('close', (code) => {
      if (code !== 0) {
        return res.status(500).json({ error: 'Error generating visualization' });
      }
      res.json({ visualization });
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
*/

// Example Python script that would generate a visualization (for reference)
/*
# generate_visualization.py
import sys
import json
from purpose.examples.visualization.rag_visualizer import RAGVisualizer

# Read the model result from stdin
model_result = sys.stdin.read()

# Create the visualizer
visualizer = RAGVisualizer(use_local_model=True)

# Generate the visualization
visualization_code = visualizer.generate_visualization(model_result)

# Output the visualization code
print(visualization_code)
*/ 