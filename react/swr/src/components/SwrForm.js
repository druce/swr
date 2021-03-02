import React, { Component } from 'react';
import { Button, Form, FormGroup, Label, Input, Col } from 'reactstrap';
import '../css/SwrForm.css';

class SwrForm extends Component {

    render() {
        return (
            <div className="row row-content">
                <div className="col-12 col-md-9">
                    <Form onSubmit={this.props.mainObj.handleSubmit}>
                    <FormGroup row>
                            <Label htmlFor="stock_alloc_pct" md={2} className="text-right">Stocks</Label>
                            <Col md={10}>
                                <Input type="text" id="stock_alloc_pct" name="stock_alloc_pct"
                                placeholder="pct"
                                value={this.props.mainObj.state.stock_alloc_pct}
                                onChange={this.props.mainObj.handleStockChange} />
                            </Col>
                        </FormGroup>
                        <FormGroup row>
                            <Label htmlFor="bond_alloc_pct" md={2} className="text-right">Bonds</Label>
                            <Col md={10}>
                                <Input type="text" id="bond_alloc_pct" name="bond_alloc_pct"
                                placeholder="pct"
                                value={this.props.mainObj.state.bond_alloc_pct}
                                onChange={this.props.mainObj.handleBondChange} />
                            </Col>
                        </FormGroup>                        
                        <FormGroup row>
                            <Label htmlFor="withdrawal_fixed_pct" md={2} className="text-right">Fixed</Label>
                            <Col md={10}>
                                <Input type="text" id="withdrawal_fixed_pct" name="withdrawal_fixed_pct"
                                placeholder="pct"
                                value={this.props.mainObj.state.withdrawal_fixed_pct}
                                onChange={this.props.mainObj.handleFixedChange} />
                            </Col>
                        </FormGroup>                        
                        <FormGroup row>
                            <Label htmlFor="withdrawal_variable_pct" md={2} className="text-right">Variable</Label>
                            <Col md={10}>
                                <Input type="text" id="withdrawal_variable_pct" name="withdrawal_variable_pct"
                                placeholder="pct"
                                value={this.props.mainObj.state.withdrawal_variable_pct}
                                onChange={this.props.mainObj.handleVariableChange} />
                            </Col>
                        </FormGroup>                        
                        <FormGroup row>
                            <Label htmlFor="withdrawal_floor_pct" md={2} className="text-right">Floor</Label>
                            <Col md={10}>
                                <Input type="text" id="withdrawal_floor_pct" name="withdrawal_floor_pct"
                                placeholder="pct"
                                value={this.props.mainObj.state.withdrawal_floor_pct}
                                onChange={this.props.mainObj.handleFloorChange} />
                            </Col>
                        </FormGroup>
                        <FormGroup row>
                            <Col md={{size: 10, offset: 2}}>
                                <Button type="submit" color="primary">
                                    Submit
                                </Button>
                            </Col>
                        </FormGroup>
                    </Form>
                </div>
            </div>
        );
    }

}
  
export default SwrForm;

