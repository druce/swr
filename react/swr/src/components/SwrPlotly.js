import React, { Component } from 'react';
import Plot from 'react-plotly.js';

class SwrPlotly extends Component {

  render() {
    return (
      <Plot
        data={[
          {
            x: this.props.x,
            y: this.props.y,
            type: 'scatter',
            mode: 'lines+markers',
            marker: {color: 'red'},
          },
          // {type: 'bar', x: this.props.mainObj.state.plot1x, y: this.props.mainObj.state.plot1y},
          {type: 'bar', x: this.props.x, y: this.props.y},
        ]}
        layout={ {width: 320, height: 240, title: 'A Fancy Plot'} }
      />
    );
  }
}

export default SwrPlotly;