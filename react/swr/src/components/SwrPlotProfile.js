// import { data } from 'jquery';
import React, { Component } from 'react';
import Plot from 'react-plotly.js';

class SwrPlotProfile extends Component {

  render() {

    let df = this.props.mainObj.state.profile_df;
    let plotly_data = [{
      x: df.data[0],
      y: df.data[1],
      name: 'Spending',
        type: 'scatter',
        mode: 'lines',
        line: {'width': 2}        
    }]; 

    return (
      <Plot
        data={plotly_data}
        layout={{width: 480,
          height: 300, 
          showlegend: false,
          title: false,
          yaxis: {title: 'Spending',
            linecolor: 'black',
            rangemode: 'tozero',
            mirror: true,
            ticks: 'inside'},
          xaxis: {title: 'Portfolio Value',
            linecolor: 'black',
            mirror: true,
            ticks: 'inside'},
          margin: {
            t: 5,
            r: 10,
            l: 50,
            b: 40,
            pad: 0
          }}
        }
      />
    );
  }
}

export default SwrPlotProfile;