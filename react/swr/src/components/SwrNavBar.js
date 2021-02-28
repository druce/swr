import React, { useState } from 'react';
import {
  Collapse,
  Navbar,
  NavbarToggler,
  NavbarBrand,
  Nav,
  NavItem,
  NavLink,
  UncontrolledDropdown,
  DropdownToggle,
  DropdownMenu,
  DropdownItem,
} from 'reactstrap';

// get our fontawesome imports
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faAddressCard, faRocket, faFolderOpen } from "@fortawesome/free-solid-svg-icons";
import '../css/SwrNavBar.css';

const SwrNavBar = (props) => {
  const [isOpen, setIsOpen] = useState(false);

  const toggle = () => setIsOpen(!isOpen);

  return (
    <div>
      <Navbar className="navbar-custom" expand="md">
        <NavbarBrand className="navbar-brand-custom" href="/">Druce.ai</NavbarBrand>
        <NavbarToggler onClick={toggle} />
        <Collapse isOpen={isOpen} navbar>
            <Nav className="ml-auto navbar-custom" navbar>
                <NavItem className="navbar-custom">
                    <NavLink href="https://druce.ai/contact.html"><FontAwesomeIcon icon={faAddressCard} /> About</NavLink>
                </NavItem>
                <UncontrolledDropdown className="navbar-custom" nav inNavbar>
                    <DropdownToggle nav caret>
                        <FontAwesomeIcon icon={faRocket} /> Projects
                    </DropdownToggle>
                    <DropdownMenu right>
                        <DropdownItem href="http://www.streeteye.com/static/Pizza/">
                        Pizza Pizza Pizza
                        </DropdownItem>
                        <DropdownItem href="http://www.streeteye.com/namegenerator/">
                        Hedge Fund Name Generator
                        </DropdownItem>
                        <DropdownItem href="http://www.streeteye.com/static/fintwit201901/">
                        FinTwit Graph
                        </DropdownItem>
                        <DropdownItem href="http://www.streeteye.com/calculator/">
                        Cat Food Calculator
                        </DropdownItem>
                    </DropdownMenu>
                </UncontrolledDropdown>
                <NavItem className="navbar-custom">
                    <NavLink href="https://druce.ai/tags.html"><FontAwesomeIcon icon={faFolderOpen} /> Archive</NavLink>
                </NavItem>
            </Nav>
        </Collapse>
      </Navbar>
    </div>
  );
}

export default SwrNavBar;