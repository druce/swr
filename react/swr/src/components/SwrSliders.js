import React, { Component } from 'react';
import { Form, FormGroup, Label, Col} from 'reactstrap';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';
import '../css/SwrSliders.css';

class SwrSliders extends Component {

    onStocksChange = value => {
        // console.log(value);
        this.props.mainObj.setState({
            stock_alloc_pct: value,
            bond_alloc_pct: 100 - value,
        }, this.props.mainObj.do_recalc);
    }

    onBondsChange = value => {
        // console.log(value);
        this.props.mainObj.setState({
            stock_alloc_pct: 100 - value,
            bond_alloc_pct: value,
        }, this.props.mainObj.do_recalc);
    }

    onFixedChange = value => {
        // console.log(value);
        this.props.mainObj.setState({
            withdrawal_fixed_pct: value
        }, this.props.mainObj.do_recalc);
    }

    onVariableChange = value => {
        // console.log(value);
        this.props.mainObj.setState({
            withdrawal_variable_pct: value

        }, this.props.mainObj.do_recalc);
    }

    onFloorChange = value => {
        // console.log(value);
        this.props.mainObj.setState({
            withdrawal_floor_pct: value
        }, this.props.mainObj.do_recalc);
    }

    render() {
        return(
            <div className="container">
            <div className="row ">
    
            <div className="col-12 col-md-9">
                <div className="col-12 col-md-8">
                    <h3 className="text-left">Asset Allocation:</h3>
                </div>
                <Form onSubmit={this.props.mainObj.handleSubmit}>
                <FormGroup row className="align-items-center">
                            <Label htmlFor="stock_alloc_pct" md={3} className="text-right">Stocks %: {this.props.mainObj.state.stock_alloc_pct}</Label>
                            <Col md={9}>
                                <Slider 
                                    value={this.props.mainObj.state.stock_alloc_pct}
                                    min={0} 
                                    max={100} 
                                    step={1}
                                    onChange={this.onStocksChange}
                                    onAfterChange={this.onStocksChange}
                                    id="stock_alloc_pct"
                                />
                            </Col>
                    </FormGroup>
                    <FormGroup row className="align-items-center">
                            <Label htmlFor="bond_alloc_pct" md={3} className="text-right">Bonds %: {this.props.mainObj.state.bond_alloc_pct}</Label>
                            <Col md={9}>
                                <Slider 
                                    value={this.props.mainObj.state.bond_alloc_pct}
                                    min={0} 
                                    max={100} 
                                    step={1}
                                    onChange={this.onBondsChange}
                                    onAfterChange={this.onBondsChange}
                                    id="bond_alloc_pct"
                                />
                            </Col>
                    </FormGroup>
                    
                    <div className="col-12 col-md-9">
                        <h3 className="text-left">Annual Withdrawals:</h3>
                    </div>

                    <FormGroup row className="align-items-center">
                            <Label htmlFor="withdrawal_fixed" md={3} className="text-right">Fixed %: {this.props.mainObj.state.withdrawal_fixed_pct}</Label>
                            <Col md={9}>
                                <Slider 
                                    value={this.props.mainObj.state.withdrawal_fixed_pct}
                                    min={-6} 
                                    max={6} 
                                    step={0.1}
                                    onChange={this.onFixedChange}
                                    onAfterChange={this.onFixedChange}
                                    id="withdrawal_fixed"
                                />
                            </Col>
                    </FormGroup>

                    <FormGroup row className="align-items-center">
                            <Label htmlFor="withdrawal_variable" md={3} className="text-right">Variable %: {this.props.mainObj.state.withdrawal_variable_pct}</Label>
                            <Col md={9}>
                                <Slider 
                                    value={this.props.mainObj.state.withdrawal_variable_pct}
                                    min={0} 
                                    max={10} 
                                    step={0.1}
                                    onChange={this.onVariableChange}
                                    onAfterChange={this.onVariableChange}
                                    id="withdrawal_variable"
                                />
                            </Col>
                    </FormGroup>

                    <FormGroup row className="align-items-center">
                            <Label htmlFor="withdrawal_floor" md={3} className="text-right">Floor %: {this.props.mainObj.state.withdrawal_floor_pct}</Label>
                            <Col md={9}>
                                <Slider 
                                    value={this.props.mainObj.state.withdrawal_floor_pct}
                                    min={0} 
                                    max={6} 
                                    step={0.1}
                                    onChange={this.onFloorChange}
                                    onAfterChange={this.onFloorChange}
                                    id="withdrawal_floor"                                    
                                />
                            </Col>
                    </FormGroup>
                    <div className="col-12 col-md-9">
                        <h3 className="text-left">Historical Outcomes:</h3>
                    </div>


                </Form>
            </div>
            </div>
            </div>
        );
    }
}

export default SwrSliders;