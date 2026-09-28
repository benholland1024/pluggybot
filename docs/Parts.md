# Parts List

Specs and sourcing for the real parts the sim is modelled on, and the sim
parameter each number feeds. Hardware honesty (PluggyPlan.md, "What stays
fixed") is why this file exists: every part is purchasable, and a sim
constant with no part behind it is a guess and is marked as one.

> **The quadruped's parts are the first sections** (#377, #378), after the
> build's bill of materials (#379), which prices them; everything after them
> is the wheeled rover's, and goes with the rover (#376 stage C). The bill
> is the `build` shelf of `protocol/parts.json`, and the table below is
> rendered from it.

> **The list is data:** `protocol/parts.json`, emitted by `uv run python -m
> pluggybot.rack.catalog` (issue #185) — every part below with its number,
> source, mass, price and the sim constant it feeds, the constant's value
> READ OFF THE MODEL when the file is built, so the fixture cannot describe a
> number the sim does not use. A number this doc does not know is `null`
> there with a reason, never a guess. The website's parts page renders it.
> This doc keeps the reasoning; the spec→parameter tables live there.

> **Sourcing note:** Pololu (US) parts are stocked by German/EU distributors — mainly [Eckstein-shop.de](https://eckstein-shop.de/Pololu_EN), plus BerryBase, EXP-Tech, Welectron, Botland and TME.eu — so no US import is needed. All prices are **approximate, incl. 19% VAT, as of July 2026** — re-check before ordering.

---

## The build: the bill of materials (#379)

Everything bought to build the quadruped, its arm, the rack, the dock and
the tools, and to build them safely: the bench rig, the gantry, spares and
shop tools. **Nothing is ordered** until #379's gate passes, and even then
the first order is one leg (#379, "The first build step"). The table is
rendered from `rack/catalog.py`'s `LINES` by `uv run python -m
pluggybot.rack.catalog`, and a test fails if the two differ.

How to read it:
- **A price is a seller's page on the day it was read**, VAT in, as a buyer
  in Germany pays; a lead time is the seller's own words. A seller quoting
  dollars or ex VAT is converted, and the line says so.
- **An allowance is a sum set aside for something not designed yet** (the
  frame, the brackets, the arm's end plate and fork, the rack, the cradle),
  never a price; its basis, under the table, says what it rests on. It
  becomes a price when the drawing gets a quote.
- **The contingency** covers shipping, the drift between reading a price
  and ordering, and the part bought twice.
- **Feeds** are the sim constants the part sets; `protocol/parts.json` has
  each one's value, read off the model when the file was built.
- **Not in it:** labour, and the budget's VAT question: a business that
  reclaims VAT spends 16 % less than the total.

<!-- bom: rendered by `uv run python -m pluggybot.rack.catalog` from rack/catalog.py's LINES; edit those, not this -->
Prices and lead times read 2026-09-28; € incl. 19% VAT; $1 = €0.8770 (ECB, 2026-09-25).

| part | qty | € each | € line | lead time | feeds |
|---|---|---|---|---|---|
| **actuation** | | | **1,894.10** | | |
| [Steadywin GIM8108-8 joint motor with the GDS68 driver (8:1, FOC, CAN, MIT mode), 48 V](https://openelab.io/products/steadywin-gim8108-8-robot-motor?variant=46976063668422) `SW-GIM8108-8-S68` | 14 | 124.95 | 1,749.30 | pre-order: none in stock, "restock in 10-20 days" (OpenELAB) | `FL_hip_abd.forcerange`, `FL_hip_abd.armature`, `legs.actuator.GIM8108_8.rated_torque`, +4 |
| [Amass XT30(2+2) connector pair: power and signal in one (male, female and pins)](https://www.3dptronics.com/electronics/xt3022-connectors-pair) `XT30(2+2)` | 16 | 5.00 /pair | 80.00 | dispatched within 2-3 days, UPS or DHL (3DPTronics, Italy) | — |
| [LAPP UNITRONIC BUS CAN 1 x 2 x 0.22 mm², shielded, by the metre](https://www.conrad.de/de/p/lapp-2170260-1-busleitung-unitronic-bus-1-x-2-x-0-22-mm-violett-meterware-603992.html) `2170260/1` | 10 | 1.98 | 19.80 | 176 m in stock, delivered on 2026-09-30 (Conrad) | — |
| [Silicone wire, 14 AWG (2.08 mm²), 1 m](https://www.mylipo.de/Silikonkabel-14AWG-208mm-schwarz_1) `SK14S / SK14R` | 20 | 2.25 | 45.00 | available now, 1-3 working days (MyLipo) | — |
| **power** | | | **344.90** | | |
| [Molicel INR21700-P45B Li-ion cell (21700, 4.5 Ah)](https://www.akkuteile.de/en/lithium-ionen-battery/size-21700/molicel/molicel-inr21700-p45b-4500mah-li-ion-battery-3-6v-3-7v_100852_3119) `INR21700-P45B` | 12 | 7.00 | 84.00 | ready to ship in 1-2 days (akkuteile.de) | `legs.model.PACK_WH`, `legs.actuator.BUS_V_NOMINAL`, `legs.actuator.BUS_V_RANGE`, +1 |
| [JBD smart BMS, 10S-17S, 120 A, Bluetooth](https://bikebattery.de/JBD-10S-17S-120A-360A-Intelligentes-Smart-Bluetooth-BMS-RS485-Android-iOS-APP-11S-12S-13S-14S-15S-16S) `10S-17S-120A-BT-UART` | 1 | 112.99 | 112.99 | short stock, 1-5 working days (bikebattery.de) | — |
| [Hilumin (nickel-plated steel) cell strip, 10 x 0.15 mm, 10 m](https://www.akkuman.de/shop/Schweissband-aus-Hilumin-10-mm-015-dick-10-m-lang) `2645314` | 1 | 19.95 | 19.95 | in stock, 1-3 working days (AKKUman) | — |
| [Mean Well DDR-60L-5 DC-DC converter, 18-75 V in, 5 V 12 A, isolated (DIN rail)](https://www.reichelt.com/de/en/dc-dc-converter-60w-5v-12a-ddr-60l-5-p256746.html) `DDR-60L-5` | 1 | 43.80 | 43.80 | in stock, 1-2 business days (Reichelt) | `legs.model.ELECTRONICS_W['compute + cameras']` |
| [Amass XT90-S anti-spark connector pair](https://www.mhm-modellbau.de/part-AM-XT90S.php) `XT90S` | 3 | 4.05 /pair | 12.15 | in stock, 1-2 working days (MHM) | — |
| [Victron MIDI fuse 60 A / 58 V](https://www.klimaworld.com/products/victron-midi-fuse-sicherungen-58v-48v-60a) `CIP133060010` | 1 | 14.29 | 14.29 | in stock, 1-3 working days (Klimaworld) | — |
| [Victron MIDI fuse holder](https://www.offgridtec.com/victron-midi-fuse-sicherung-halter.html) `CIP000050001` | 1 | 10.12 | 10.12 | ships in 1-3 working days (Offgridtec) | — |
| [Silicone wire, 10 AWG (6 mm²), 1 m](https://www.modellbau-skeries.de/p/silikonkabel-10awg-6mm-x-1000mm-rot) `2392 (red), 2393 (black)` | 4 | 4.40 | 17.60 | in stock, 1-3 days (Skeries) | — |
| The pack's printed case, fish paper, heat-shrink and 12S balance lead | 1 | allowance | 30.00 | unknown: nothing drawn to quote | `legs.model.CHOSEN.belly_depth` |
| **compute** | | | **327.55** | | |
| [Raspberry Pi 5, 8 GB](https://www.berrybase.de/raspberry-pi-5-8gb-ram) `SC1112` | 1 | 184.90 | 184.90 | in stock, 1-3 days (BerryBase) | `legs.model.ELECTRONICS_W['compute + cameras']` |
| [Raspberry Pi Active Cooler for the Pi 5](https://www.berrybase.de/raspberry-pi-active-cooler-luefter-fuer-raspberry-pi-5) `SC1148` | 1 | 5.90 | 5.90 | in stock, 1-3 days (BerryBase) | — |
| [Raspberry Pi SSD Kit, 256 GB (M.2 HAT+ and NVMe)](https://www.berrybase.de/raspberry-pi-ssd-kit-fuer-raspberry-pi-5-256gb) `SC1675` | 1 | 54.90 | 54.90 | not deliverable now (BerryBase); the 512 GB kit is in stock at Reichelt for €166.10 | — |
| [Waveshare 2-CH CAN HAT+ (MCP2515, isolated) for the Raspberry Pi](https://eckstein-shop.de/WaveShare-2CH-Isolated-CAN-Bus-Expansion-HAT-for-Raspberry-Pi) `27338` | 1 | 29.95 | 29.95 | in stock, 1-3 days (Eckstein) | — |
| [Waveshare 2-CH CAN FD HAT (MCP2518FD, isolated) for the Raspberry Pi](https://www.welectron.com/Waveshare-17075-2-CH-CAN-FD-HAT_1) `17075` | 1 | 51.90 | 51.90 | in stock (short supply), 1-3 business days (Welectron) | — |
| **sensors** | | | **491.80** | | |
| [Slamtec RPLIDAR C1 2D LIDAR](https://funduinoshop.com/en/electronic-modules/sensors/movement-distance/rplidar-c1-dtof-lidar-3600-laser-range-scanner-12m-ip54-slamtec) `RPLIDAR C1` | 1 | 84.90 | 84.90 | 1-3 business days (Funduino) | `legs.model.ELECTRONICS_W['lidar']`, `legs.model.MassBudget.scanner`, `lidar.pos[2]`, +2 |
| [RealSense D435 depth camera (active IR stereo)](https://www.digikey.de/de/products/detail/realsense/82635AWGDVKPRQ/9926002) `D435` | 1 | 340.01 | 340.01 | none in stock, the maker's standard lead 14 weeks (DigiKey DE); 21 days at MyBotShop for €419.99 | `depth_eye.fovy`, `legs.model.ELECTRONICS_W['depth camera']`, `legs.model.MassBudget.depth_cam`, +2 |
| [Raspberry Pi Camera Module 3](https://www.berrybase.de/raspberry-pi-camera-module-3-12mp) `Camera Module 3` | 1 | 28.90 | 28.90 | in stock, 1-3 days (BerryBase) | `nav_eye.fovy`, `legs.model.MassBudget.cameras` |
| [Raspberry Pi camera cable, standard to mini, 200 mm](https://www.berrybase.de/raspberry-pi-camera-cable-standard-mini-200mm) `SC1128` | 1 | 1.20 | 1.20 | in stock, 1-3 days (BerryBase) | — |
| [TDK InvenSense ICM-42688-P evaluation board (IMU)](https://www.digikey.de/de/products/detail/tdk-invensense/EV-ICM-42688-P/18634550) `EV_ICM-42688-P` | 1 | 36.79 | 36.79 | 138 in stock, the maker's standard lead 16 weeks (DigiKey DE) | `perception.imu.GYRO_NOISE`, `perception.imu.ACCEL_NOISE`, `perception.imu.GYRO_SCALE` |
| **frame and legs** | | | **1,229.60** | | |
| The torso's frame: two 3 mm aluminium side plates, end caps and cross members | 1 | allowance | 300.00 | unknown: nothing drawn to quote | `legs.model.MassBudget.frame`, `legs.model.CHOSEN.torso` |
| [Carbon fibre tube, roll-wrapped, 25 mm OD / 22 mm ID, 1 m (Easy Composites)](https://www.easycomposites.eu/25mm-23mm-woven-finish-carbon-fibre-tube) `CFT-WF-25-22-1` | 2 | 46.05 | 92.10 | more than 50 in stock, dispatched the same day before 13:00, DHL 2 days (Easy Composites, NL) | `legs.model.CHOSEN.thigh`, `legs.model.CHOSEN.shank`, `legs.model.CHOSEN.thigh_mass`, +1 |
| The legs' motor mounts, hip brackets, knee housings and tube clamps, with their fasteners | 1 | allowance | 800.00 | unknown: nothing drawn to quote | `legs.model.CHOSEN.hip_out` |
| [Dunlop Pro squash balls, box of 12 (40 mm, 24 g each)](https://www.squashpoint.de/dunlop-pro-squashball-12er-box.html) `700108` | 1 | 37.50 | 37.50 | in stock, 1-3 working days (squashpoint.de) | `legs.model.CHOSEN.foot_r` |
| **arm** | | | **270.56** | | |
| [Carbon fibre tube, roll-wrapped, 20 mm OD / 18 mm ID, 1 m (Easy Composites)](https://www.easycomposites.eu/20mm-woven-finish-carbon-fibre-tube) `CFT-WF-20-18-1` | 1 | 27.91 | 27.91 | more than 50 in stock, dispatched the same day before 13:00, DHL 2 days (Easy Composites, NL) | `legs.arm.ArmSpec.upper`, `legs.arm.ArmSpec.fore`, `legs.arm.TUBE_R` |
| [SKF 61800-2RS1 deep-groove ball bearing, 10 x 19 x 5 mm](https://www.kugellager-shop.net/61800-2rs1-skf-rillenkugellager-10x19x5mm.html) `61800-2RS1` | 4 | 7.60 | 30.40 | in stock, 1-4 days (kugellager-shop.net) | — |
| [igus igubal KCRM-05 rod end, M5 female, right-hand](https://www.igus.de/product/igubal_KCRM_KCLM?artNr=KCRM-05) `KCRM-05` | 6 | 5.43 | 32.58 | ready to ship in 24 hours (igus) | `legs.arm.ArmSpec.rods_mass` |
| The arm's end plate, fork and lean-pad (two prongs, four 60° V's, two 53° end-ramps), and the parallelogram's three rods | 1 | allowance | 150.00 | unknown: nothing drawn to quote | `legs.arm.ArmSpec.plate_mass`, `legs.arm.ForkSpec.flank_deg`, `legs.arm.ForkSpec.fork_y`, +1 |
| [PTFE glass-fabric adhesive tape, 0.13 mm, 25 mm x 30 m](https://shop.hightechflon.com/PTFE-Teflon-Klebeband-0-13%20mm-ptfe-glasgewebe/Page-16-1-96-183.aspx) `PTFE Klebeband 0.13 SW` | 1 | 29.67 | 29.67 | 2-4 working days (High-tech-flon) | `legs.arm.RAMP_MU` |
| **tools** | | | **158.95** | | |
| Tool peg, 220 mm: two 6 mm steel conductors on an insulating bush | 4 | allowance | 30.00 | unknown: nothing drawn to quote | `legs.rack.PEG_HALF`, `legs.rack.PEG_MASS` |
| [FEETECH FS90MG micro servo (digital, metal gears)](https://eckstein-shop.de/Feetech-FS90MG-6V-22kgcm-Digital-Servo) `FS90MG` | 2 | 5.95 | 11.90 | unknown: not read for this bill; the price is the workshop catalog's (#199, 2026-09-14) | — |
| [Actuonix L12-100-50-6-R micro linear servo (100 mm stroke, 50:1, 6 V, RC input)](https://www.digikey.de/de/products/detail/actuonix-motion-devices-inc/L12-100-50-6-R/11689540) `L12-100-50-6-R` | 1 | 77.05 | 77.05 | unknown: not read for this bill; the price is the workshop catalog's (#199, 2026-09-14) | — |
| ESP32-class board, one per module | 4 | 5.00 | 20.00 | unknown: no board chosen | `power.MODULE_IDLE_W` |
| Small SPI/I2C display driven by the module's ESP32 | 1 | allowance | 20.00 | unknown: no display chosen | `rack.coupling.LCD_SCREEN_HALF` |
| **rack** | | | **108.69** | | |
| [Bay presence switch: one Omron D2F-01L2 hinge-roller-lever microswitch in each tool bay's V-tray](https://www.digikey.de/de/products/detail/omron-electronics-inc-emc-div/D2F-01L2/368444) `D2F-01L2` | 3 | 2.59 | 7.77 | unknown: not read for this bill; the price is the workshop catalog's (#199, 2026-09-14) | `bay*_tray_l_*` |
| ESP32-class board, one per module | 1 | 5.00 | 5.00 | unknown: no board chosen | `power.MODULE_IDLE_W` |
| [Aluminium extrusion 30x30 light, B-type slot 8, cut to length (1 m)](https://www.dold-mechatronik.de/Aluminum-Profile-30x30L-B-Type-Groove-8-084kg-m-Customized-Cutting-50-to-6000mm) `67700-Z` | 4 | 10.40 | 41.60 | in stock, 4-5 working days (Dold) | `legs.rack.RackSpec.peg_z`, `legs.rack.RackSpec.rail_z` |
| [Angle bracket 30, B-type slot 8, with its screws and hammer nuts](https://www.dold-mechatronik.de/Angle-30-B-type-groove-8-with-mounting-kit-and-cap) `66918-BSA` | 8 | 1.79 | 14.32 | in stock, 3-4 working days (Dold) | — |
| The rack's back board, its six printed V-trays, and the rack's and the dock's printed tags | 1 | allowance | 40.00 | unknown: a board from a DIY store; the trays and tags printed in the shop | `legs.rack.TRAY_Y`, `legs.rack.RACK_TAG_IDS`, `legs.dock.DockSpec.board_x` |
| **dock** | | | **262.98** | | |
| [Mill-Max 0858-0-15-20-82-14-11-0 spring-loaded pin (gold, 1.27 mm plunger, 2.29 mm stroke)](https://www.digikey.de/de/products/detail/mill-max-manufacturing-corp/0858-0-15-20-82-14-11-0/7667985) `0858-0-15-20-82-14-11-0` | 4 | 2.12 | 8.48 | 18,820 in stock, the maker's standard lead 4 weeks (DigiKey DE) | `legs.dock.PIN_TRAVEL`, `legs.dock.PIN_STROKE`, `legs.dock.PIN_TIP_R`, +2 |
| [Belly pads: 2-layer ENIG PCB, 160 x 20 mm, a batch of 6 (AISLER Budget)](https://aisler.net/en/products/boards) | 1 | 38.83 | 38.83 | dispatched in about 10 business days, then 2 days by post (AISLER) | `legs.model.PAD_HALF`, `legs.model.PAD_Y` |
| [UHMW-PE (PE 1000) sheet, natural, 300 x 300 x 10 mm](https://www.kunststoffhaus.de/0020-RI45070-DIV-10-25-mm-PE-UHMW-Platte-PE-1000-Polyethylen-Abm-und-Farbe-waehlbar) `0100-RI45070-0300-0300-N` | 1 | 46.67 | 46.67 | available now, 3-5 working days (kunststoffhaus.de) | `legs.dock.DockSpec.cradle_mu` |
| The dock's cradle: the bed and funnel faces cut from the sheet, on a base | 1 | allowance | 100.00 | unknown: nothing drawn to quote | `legs.dock.DockSpec.mouth_half`, `legs.dock.DockSpec.bed_half` |
| [NOEIFEVO 50.4 V 5 A Li-ion charger (12S, CC/CV)](https://www.noeifevo.de/products/noeifevo-50-4v-5a-lithium-ladegerat-fur-12s-44-4v-li-ionen-lipo-akku-e-bike-roller-ladegerat-led-anzeige-aluminiumgehause) `HRH-300L-50405-XT60` | 1 | 69.00 | 69.00 | 3-7 working days from its Polish warehouse, 7-15 from China; the page does not say which | `legs.dock.CHARGE_A`, `legs.dock.CHARGE_W` |
| **safety** | | | **322.53** | | |
| [Albright SW80B contactor, 12 V coil, with blowouts (100 A, 96 V DC)](https://www.rijregeling.nl/en/product/albright-contactors-and-accessories/sw80-series/sw80b549-albright-contactor-12v-contactor/) `SW80B549` | 1 | 82.28 | 82.28 | unknown: the page states none (it can be put in the cart) | — |
| [Schneider Harmony XB5AS8442 emergency stop, 40 mm twist-release head, 1 NC](https://www.reichelt.com/de/en/shop/product/emergency_stop_switch_harmony_xb5_twist_unlock_22_mm_1_nc-382385) `XB5AS8442` | 2 | 30.95 | 61.90 | available on 2026-10-01 (Reichelt) | — |
| [RadioMaster Pocket ELRS transmitter (EU LBT)](https://n-factory.de/RadioMaster-Pocket-ELRS-Remote-Control-EU-LBT) `Pocket ELRS EU-LBT` | 1 | 79.95 | 79.95 | ships the same day (n-Factory) | — |
| [RadioMaster ER6 ELRS PWM receiver (LBT)](https://n-factory.de/RadioMaster-ER6-ELRS-LBT-24-GHz-PWM-Receiver) `ER6 ELRS LBT` | 1 | 31.95 | 31.95 | ships the same day (n-Factory) | — |
| [Pololu RC Switch with Relay (assembled)](https://eckstein-shop.de/Pololu-RC-Switch-with-Relay-Assembled-EN) `2804` | 1 | 21.36 | 21.36 | about 8-10 days (Eckstein) | — |
| [Molicel INR18650-P28A Li-ion cell (18650, flat top)](https://www.lipo24.de/products/molicel-inr18650-p28a-2800mah-35a-lithium-ionen-akku-3-6v-3-7v) `INR18650-P28A` | 2 | 5.95 | 11.90 | available, 1-3 days (LiPo24) | — |
| [Mean Well DDR-30L-12 DC-DC converter, 18-75 V in, 12 V 2.5 A, isolated (DIN rail)](https://www.reichelt.com/de/en/shop/product/dc_dc_converter_30w_12v_2_5a-256735) `DDR-30L-12` | 1 | 27.10 | 27.10 | in stock, 1-2 business days (Reichelt) | — |
| [Arcol HS25 wirewound resistor, 47 Ω, 25 W, aluminium housing](https://www.reichelt.com/de/en/shop/product/wirewound_resistor_axial_25_w_47_ohm_1_-233471) `HS25 47R F` | 1 | 2.99 | 2.99 | in stock, 1-2 business days (Reichelt) | — |
| [Hongfa HF115F relay, 12 V coil, 16 A changeover](https://www.reichelt.com/de/en/shop/product/miniature_power_relay_12_v_dc_16_a_1_changeover_contact-260113) `HF115F-012-1ZS3A` | 1 | 3.10 | 3.10 | in stock, 1-2 business days (Reichelt) | — |
| **bench leg** | | | **422.33** | | |
| [Korad KA6005P lab power supply, 0-60 V, 0-5 A, safety terminals](https://www.welectron.com/Korad-KA6005P-Benchtop-Power-Supply-Safety-Terminals) `KA6005P` | 1 | 239.00 | 239.00 | in stock, 1-3 business days (Welectron) | — |
| [UCAN V1.0 USB-CAN adapter (candleLight firmware)](https://eckstein-shop.de/ucan-v1-usb-can-adapter) `OP00014` | 1 | 8.99 | 8.99 | in stock, 1-3 days (Eckstein) | — |
| [Linear rail MGN12, 1000 mm (Dold)](https://www.dold-mechatronik.de/Linearfuehrung-MGN12R-1000mm) `MGN12R-1000` | 1 | 47.00 | 47.00 | in stock, 3-4 working days (Dold) | — |
| [Linear carriage MGN12H (Dold)](https://www.dold-mechatronik.de/Linearwagen-MGN12H) `MGN12H` | 1 | 13.00 | 13.00 | in stock, 3-4 working days (Dold) | — |
| [Aluminium extrusion 30x30 light, B-type slot 8, cut to length (1 m)](https://www.dold-mechatronik.de/Aluminum-Profile-30x30L-B-Type-Groove-8-084kg-m-Customized-Cutting-50-to-6000mm) `67700-Z` | 3 | 10.40 | 31.20 | in stock, 4-5 working days (Dold) | `legs.rack.RackSpec.peg_z`, `legs.rack.RackSpec.rail_z` |
| [Angle bracket 30, B-type slot 8, with its screws and hammer nuts](https://www.dold-mechatronik.de/Angle-30-B-type-groove-8-with-mounting-kit-and-cap) `66918-BSA` | 6 | 1.79 | 10.74 | in stock, 3-4 working days (Dold) | — |
| The bench leg's mount: the carriage's plate holding the hip, and the hop's end stops | 1 | allowance | 60.00 | unknown: nothing drawn to quote | `legs.model.CHOSEN.stand_height` |
| [Raspberry Pi 27 W USB-C power supply (EU)](https://www.berrybase.de/raspberry-pi-27w-usb-c-power-supply-netzteil-weiss) `SC1152` | 1 | 12.40 | 12.40 | in stock, 1-3 days (BerryBase) | — |
| **gantry** | | | **273.68** | | |
| [Aluminium extrusion 40x40 light, I-type slot 8, cut to length (1 m)](https://www.dold-mechatronik.de/Aluminum-Profile-40x40L-I-Type-Groove-8-176kg-m-Customized-Cutting-50-to-6000mm) `60800-Z` | 8 | 18.90 | 151.20 | in stock, 4-5 working days (Dold) | — |
| [Dönges rope ratchet, 113 kg, 5 m](https://www.feuerwehrdiscount.de/seilzugratsche-rope-ratchet/auslastung-bis-max.-68-kg) `Seilzugratsche 113 kg` | 1 | 33.99 | 33.99 | in stock, 1-2 working days (feuerwehrdiscount.de) | — |
| [Mammut Magic Sling 12.0, 120 cm (22 kN)](https://www.bergfreunde.de/mammut-magic-sling-120-bandschlinge/) `314-0107` | 2 | 18.00 | 36.00 | 2-3 working days (Bergfreunde) | — |
| [Bungee cord, 8 mm, 10 m (Hummelt)](https://hummelt-shop.de/produkt/expanderseil-gummiseil-8mm-blau/) `1037` | 1 | 12.49 | 12.49 | in stock, 2-3 days (Hummelt) | — |
| The gantry's fittings: angle brackets and T-nuts for 40x40 slot 8, and two eye bolts | 1 | allowance | 40.00 | unknown: not read for 40x40 | — |
| **spares** | | | **320.40** | | |
| [Steadywin GIM8108-8 joint motor with the GDS68 driver (8:1, FOC, CAN, MIT mode), 48 V](https://openelab.io/products/steadywin-gim8108-8-robot-motor?variant=46976063668422) `SW-GIM8108-8-S68` | 2 | 124.95 | 249.90 | pre-order: none in stock, "restock in 10-20 days" (OpenELAB) | `FL_hip_abd.forcerange`, `FL_hip_abd.armature`, `legs.actuator.GIM8108_8.rated_torque`, +4 |
| [Molicel INR21700-P45B Li-ion cell (21700, 4.5 Ah)](https://www.akkuteile.de/en/lithium-ionen-battery/size-21700/molicel/molicel-inr21700-p45b-4500mah-li-ion-battery-3-6v-3-7v_100852_3119) `INR21700-P45B` | 2 | 7.00 | 14.00 | ready to ship in 1-2 days (akkuteile.de) | `legs.model.PACK_WH`, `legs.actuator.BUS_V_NOMINAL`, `legs.actuator.BUS_V_RANGE`, +1 |
| [Victron MIDI fuse 60 A / 58 V](https://www.klimaworld.com/products/victron-midi-fuse-sicherungen-58v-48v-60a) `CIP133060010` | 2 | 14.29 | 28.58 | in stock, 1-3 working days (Klimaworld) | — |
| [Mill-Max 0858-0-15-20-82-14-11-0 spring-loaded pin (gold, 1.27 mm plunger, 2.29 mm stroke)](https://www.digikey.de/de/products/detail/mill-max-manufacturing-corp/0858-0-15-20-82-14-11-0/7667985) `0858-0-15-20-82-14-11-0` | 6 | 2.12 | 12.72 | 18,820 in stock, the maker's standard lead 4 weeks (DigiKey DE) | `legs.dock.PIN_TRAVEL`, `legs.dock.PIN_STROKE`, `legs.dock.PIN_TIP_R`, +2 |
| [SKF 61800-2RS1 deep-groove ball bearing, 10 x 19 x 5 mm](https://www.kugellager-shop.net/61800-2rs1-skf-rillenkugellager-10x19x5mm.html) `61800-2RS1` | 2 | 7.60 | 15.20 | in stock, 1-4 days (kugellager-shop.net) | — |
| **shop** | | | **1,119.82** | | |
| [keenlab kWeld spot welder, complete kit, cables assembled](https://www.keenlab.de/index.php/product/kweld-complete-kit/) `kWeld complete kit` | 1 | 210.63 | 210.63 | 28 in stock (keenlab) | — |
| [keenlab kCap ultracapacitor module for the kWeld](https://www.keenlab.de/index.php/product/kweld-ultracapacitor-module/) `kCap` | 1 | 153.51 | 153.51 | 41 in stock (keenlab) | — |
| [Engineer PA-09 crimping tool (JST-PH/GH class, AWG 32-20)](https://www.kiwi-electronics.com/en/jst-crimping-tool-pa-09-916) `PA-09` | 1 | 47.59 | 47.59 | 31 in stock, ships the same day (Kiwi, NL) | — |
| [Knipex 97 62 145 A ferrule crimping pliers, 0.25-2.5 mm²](https://www.reichelt.com/de/en/shop/product/crimping_pliers_for_end_sleeves_ferrules_-184328) `97 62 145 A` | 1 | 30.80 | 30.80 | in stock, 1-2 business days (Reichelt) | — |
| [Pine64 Pinecil V2 soldering iron](https://eleshop.de/pinecil-smart-mini-tragbarer-lotkolben.html) `PINECIL-BB2` | 1 | 39.90 | 39.90 | in stock, ships the same day (eleshop) | — |
| [UNI-T UT161E true-RMS multimeter](https://eleshop.de/uni-t-ut161e.html) `UT161E` | 1 | 95.60 | 95.60 | expected on 2026-10-07 (eleshop) | — |
| [Proxxon MicroClick MC 5 torque wrench, 1-5 N·m, 1/4"](https://www.reichelt.com/de/en/shop/product/micro-click_torque_screwdriver_5_s-91768) `23347` | 1 | 77.99 | 77.99 | in stock, 1-2 business days (Reichelt) | — |
| [Wera 950 PKS/9 SM N metric hex key set, 1.5-10 mm](https://www.reichelt.com/de/en/shop/product/spanner_set_hex_950_pks_smn_9-pieces-169289) `05133163001` | 1 | 21.80 | 21.80 | in stock, 1-2 business days (Reichelt) | — |
| [Bambu Lab P1S 3D printer (printer only)](https://eu.store.bambulab.com/products/p1s) `P1S` | 1 | 379.00 | 379.00 | ships from the EU warehouse in 1-3 business days (Bambu Lab) | — |
| [PETG filament, 1.75 mm, 1 kg (DAS FILAMENT)](https://dasfilament.de/produkt/petg-filament-175-mm-schwarz-1-kg/) `F10765` | 3 | 21.00 | 63.00 | unknown: in stock; the page and its terms give no delivery time | — |

| | € | $ |
|---|---|---|
| priced lines | 5,977.89 | |
| allowances (nothing designed yet) | 1,570.00 | |
| contingency, 15% | 1,132.18 | |
| **total** | **8,680.07** | **9,897.88** |
| budget | 17,539.24 | 20,000.00 |

**The allowances**, each a sum set aside, never a price:
- **The pack's printed case, fish paper, heat-shrink and 12S balance lead** (power), €30.00: the case printed in the shop's PETG; fish paper, heat-shrink and a 12S balance lead
- **The torso's frame: two 3 mm aluminium side plates, end caps and cross members** (frame and legs), €300.00: six to eight 3 mm AlMg3 parts: AluFritze's online configurator prices a plain 300 x 150 mm rectangle at €20.02 (four for €80.10, 10-14 working days); a plate with holes is priced from its drawing
- **The legs' motor mounts, hip brackets, knee housings and tube clamps, with their fasteners** (frame and legs), €800.00: twelve motor mounts, four hip brackets and eight tube clamps, nothing drawn, sized as machined aluminium at a job shop; printed on the shop's P1S they cost their filament
- **The arm's end plate, fork and lean-pad (two prongs, four 60° V's, two 53° end-ramps), and the parallelogram's three rods** (arm), €150.00: one machined aluminium plate with its prongs, V's and ramps, and three 5 mm rods threaded M5: a plain 4 mm plate is €23.12 at AluFritze, the V's and ramps are machining
- **Tool peg, 220 mm: two 6 mm steel conductors on an insulating bush** (tools), €30.00: four pegs cut and turned from a metre of 6 mm silver steel and an acetal bush each, as the rover's peg (`peg_rod_6mm`)
- **Small SPI/I2C display driven by the module's ESP32** (tools), €20.00: a small SPI display for the LCD tool's ESP32; the rover's LCD module never chose one
- **The rack's back board, its six printed V-trays, and the rack's and the dock's printed tags** (rack), €40.00: a 1.0 x 0.6 m plywood back board, six V-trays printed in PETG from the shop's spools, and the rack's six and the dock's four 60 mm tags printed and laminated
- **The dock's cradle: the bed and funnel faces cut from the sheet, on a base** (dock), €100.00: the bed and the funnel's faces routed from the sheet and screwed to a base: an hour at a job shop, or by hand
- **The bench leg's mount: the carriage's plate holding the hip, and the hop's end stops** (bench leg), €60.00: a 4 mm AlMg3 plate at AluFritze (a plain 300 x 150 mm one is €23.12) and printed end stops
- **The gantry's fittings: angle brackets and T-nuts for 40x40 slot 8, and two eye bolts** (gantry), €40.00: eight angle brackets with T-nuts for 40x40 slot 8 and two eye bolts; the 30x30's brackets are €1.79-2.20 at Dold, the 40x40's were not read
<!-- /bom -->

**What the bill decided** (each part's entry in `rack/catalog.py` carries
the rest):
- **The pack is built here.** No German seller publishes a 12S1P built to
  order (the closest is a 12S3P at several times the price, in 2–4 weeks),
  so the bill buys the cells, a BMS, cell strip and a spot welder. The BMS
  is a 120 A one because the 60 A and 80 A versions were sold out.
- **The e-stop cuts the motors, not the computer.** A contactor on the
  motor bus holds closed only while its coil circuit runs through the
  button on the robot AND the wireless relay; the Pi stays up to log what
  happened. A pre-charge resistor fills the drivers' capacitors before the
  contactor closes. The wireless half is a hobby RC link whose receiver's
  failsafe opens the relay, at a fifth of a commercial wireless e-stop.
- **Four CAN buses on the Pi, through two stacked HATs** (one a pair of
  legs, the arm on the third); four USB adapters are the fallback.
- **The bench leg hops on the pack.** The lab supply is for current-limited
  first power-ups: it cannot take back what the motors return on landing.
- **Hardware the bill does not choose yet** is an allowance, not a guess at
  a price: the frame, the legs' structure, the arm's end plate and fork,
  the rack, the cradle, the tools' pegs.

**Watch before ordering:**
- **The GIM8108-8 is on pre-order** (restock stated as 10–20 days, no
  count); sixteen go through the seller's quote form.
- **The D435 has the longest lead** (DigiKey's maker lead is 14 weeks; one
  German reseller quotes 21 days at a higher price), and RealSense is being
  sold to Cognex (announced 2026-09-22).
- **The contactor's coil must be continuous-rated**; the seller does not say
  which coil class it sells.
- **Open in #379:** the computer's budget (the bill carries the rover's Pi 5
  8 GB; the 16 GB is in stock), the tools' supply voltage at the peg, and
  every drawing an allowance waits on.

---

## The quadruped body (#377)

Chosen 2026-09-26 by `scripts/quad_spike.py`; the tables that chose it are
SimNotes, "The quadruped body", the body is `pluggybot.legs.model.CHOSEN`
(`models/quadruped.xml`), and every actuator number sits at its constant in
`legs/actuator.py` with its source. **Nothing is ordered**: #379 holds the
bill of materials and the gate before the first purchase.

| part | chosen | mass | price | what it feeds |
|---|---|---|---|---|
| leg actuator ×12 | Steadywin **GIM8108-8** with the GDS68 driver (FOC, CAN, MIT mode) | 396 g, Ø97 × 55 mm | €124.95 (OpenELAB, Munich); $129.20 (Steadywin) | `actuator.GIM8108_8` |
| pack | **12S1P Molicel P45B** 21700 (43.2 V nominal, 36–50.4 V, 194 Wh) | 0.95 kg: twelve 70 g cells plus a BMS, case and leads (estimated) | built here: the cells, BMS and strip are in the bill above | `MassBudget.battery`, `actuator.BUS_V_RANGE` |
| 2D LIDAR | Slamtec RPLIDAR C1 (the rover's) | 110 g | as the rover's | 1.15 W, the maker's 230 mA at 5 V |
| depth camera | RealSense D435 (the rover's) | 75 g (datasheet, March 2026) | as the rover's | 2.0 W, as `power.DEPTH_CAMERA_W` |
| compute, two cameras, IMU | as the rover's (Raspberry Pi 5, Camera Module 3) | in `electronics` | | 6.0 W, the rover's figure less its LIDAR |
| frame | aluminium side plates and cross members | 1.0 kg (estimated) | `null` — no design yet | `MassBudget.frame` |
| thigh, shank, foot | tube, no belt (the knee is direct), rubber foot | 0.21 kg a leg (estimated) | `null` — no design yet | `BodySpec.*_mass` |
| #378's arm | chosen by #378 ("The arm and its coupling", below); the body is still flown at the placeholder | 1.17 kg (the placeholder budgeted 0.9) + a tool up to 0.40 kg | — | the legs' torques re-flown with it and a tool aboard: the knee's p99.5 ≤ 9.4 N·m on the flat, 14.0 up the house's flight (13.5 without it) |

**Why 48 V (12S).** The GIM8108-8 is rated at 48 V, and joint SPEED, not
torque, is what binds this body: a 1.5 m/s trot drives a joint to 90 % of
its no-load speed on a nominal pack and saturates it on an empty one. At
24 V the no-load speed is about 20 rad/s instead of 31 (the GDS sheet's
curve), which would cap the robot near a walk.

**What the datasheets leave open, and what the sim does about it:**
- **Steadywin's two selection tables disagree** (the GDS and GDZ .xlsx on
  the store page: rated torque 7.5 vs 6.71 N·m, Kt 1.0 vs 1.19, 110 vs 256
  rpm rated). The sim takes the lower figure, and the loss model the one
  pair (Kt 1.19, R 0.72) whose 1.5·R·I² fits under the maker's own 48 V
  efficiency curve. A written answer from Steadywin would settle it.
- **Rotor inertia is not published.** The one transcribed value (4.55e-6
  kg·m²) is 10–25× under every motor of the class; the sim uses the class's
  range, 4.1e-5 to 1.1e-4 (training samples it), nominal 7.2e-5 (the Mini
  Cheetah actuator's, the same 21 pole pairs).
- **Heat:** no Steadywin part publishes a thermal resistance; the sim uses
  the Mini Cheetah actuator's measured 1.23 K/W and 32 J/K (Katz 2018), up
  to the GDS68's 90 °C motor alarm.
- **Friction, command latency and backlash** are unpublished; training
  randomises 0.05–0.6 N·m and 0–20 ms, and backlash (15 arcmin in the
  tables) is encoder noise to the policy.

**The body that was too small.** A Pupper-v3-class body (3 kg, GIM4305-10,
1.0 N·m continuous) stands and walks as built, but carrying the suite
(4.4 kg) it spends 92 % of its continuous torque STANDING, is 23 % over it
holding the arm out, cannot push itself up from its belly or up a 0.12 m
curb, and its whole leg (0.17 m) is shorter than a house's riser.

### The IMU and the encoders, as the robot reads them (#386)

Both bodies carry the same IMU, and neither is told the sim's exact angles
or rates: `perception/imu.py` and `perception/encoders.py` give every
reading the part's own error, each number from its sheet or a stated range.

| part | what the sim adds | from |
|---|---|---|
| IMU: TDK **ICM-42688-P** | gyro 0.0028 °/s/√Hz white noise; accelerometer 65 µg/√Hz (x, y), 70 (z) | DS-000347 rev 1.9, Tables 1–2 (tested in production) |
| | the offset left after the boot calibration: ±0.005 °/s/°C (gyro) and ±0.15 mg/°C (accel) over `TEMP_SWING_C` = 10 °C — ±0.05 °/s and ±1.5 mg, drawn per axis | the sheet's variation over temperature (characterised); the 10 °C swing is a stated choice. The initial ±0.5 °/s and ±20 mg are what the calibration removes |
| | scale ±0.5 % per axis, gyro and accelerometer | the sheet's initial tolerance (tested in production) |
| | not modelled: cross-axis (±1.25 % gyro, ±1 % accel), nonlinearity (±0.1 %) | small against rates that average out on a near-level body |
| rover wheel encoders | whole counts, 3200 a wheel turn | Pololu #4753 (64 CPR × 50) |
| leg joints (GDS68, MIT mode) | position in 16 bits over ±12.5 rad (0.38 mrad), velocity in 12 bits over ±65 rad/s (32 mrad/s), on top of the gearbox backlash | the MIT protocol's fields; the ranges are the driver's settings, unpublished — stated as the protocol's defaults in Katz's firmware, the velocity range above the joint's 31.4 rad/s no-load speed |

### The dock (#378)

The robot charges by lying down onto a cradle (`pluggybot.legs.dock`, flown
by `scripts/dock_spike.py`); the tables that chose it are SimNotes, "The
quadruped's dock". Nothing is ordered (#379).

| part | chosen | mass | price | what it feeds |
|---|---|---|---|---|
| charge contacts ×4 | Mill-Max **0858-0-15-20-82-14-11-0** spring-loaded pin, two a pole: 12 A at a 30 °C rise (9.6 A derated), 20 mΩ max, 25 g free and 120 g at its rated 1.143 mm of a 2.286 mm stroke, a 1.27 mm plunger, gold over nickel | under 1 g each | €2.50 each, €2.12 from ten (DigiKey DE, 2026-09-28) | `dock.PIN_*`, `dock.pole_spring()` |
| belly pads ×2 | 160 × 20 mm plated strips flush with the pack's underside, 60 mm apart | in the pack's 0.95 kg | `null` — no design yet (a gold-finished PCB strip is the likely part) | `model.PAD_HALF`, `model.PAD_Y` |
| charger | a 12S Li-ion CC-CV charger, 50.4 V at 5 A (NOEIFEVO's in the bill; BOUNDMOTOR's is the other) | not published | €69 (NOEIFEVO); $99 plus duty (BOUNDMOTOR) | `dock.CHARGE_A`; `CHARGE_W`, 216 W at the pack's nominal 43.2 V |
| cradle: bed and funnel faces | machined UHMW-PE: sliding friction 0.12–0.17 (a supplier's figure), polyethylene on steel 0.2 static (Engineering ToolBox) | `null` — no design yet | `null` — no design yet | `DockSpec.cradle_mu`, flown at 0.3 |
| tag board | four 60 mm tag36h11 tags (ids 25–28) printed on a board on a post | — | — | `tags.DOCK_TAG_IDS`, `DOCK_TAG_SIZE`, `dock.tag_layout` |

**Why 5 A.** The Molicel P45B datasheet gives 4.5 A as its standard charge
(1.5 h) and 13.5 A as its maximum (70 °C cut-off). 5 A is 1.1C for the 12S1P
pack; one pin a pole carries it with 4.6 A to spare, and the four lose half
a watt.

**Why sprung pins, and not the robot's weight.** The dock was proposed with
the robot's own weight (~90 N) as the contacts' preload. But a lying
quadruped rests on its belly AND on four limp legs, so rigid contacts under
it carry whatever the legs leave them: a robot lying 6 mm off centre put
all 57 N through one bar and hung the other pad 0.3 mm clear. The weight
seats the belly on the bed; the pins' springs, 2.35 N a pole at their
rated travel, are the contact.

### The arm and its coupling (#378)

Two pitch joints on the torso's top front, both motors at the shoulder, a
passive parallelogram keeping the end plate level (`pluggybot.legs.arm`),
and a rack the robot walks up to (`pluggybot.legs.rack`), flown by
`scripts/arm_spike.py`; the tables that chose them are SimNotes, "The
quadruped's arm, its coupling and the rack". Nothing is ordered (#379), and
nothing here is in the served world yet (#375, step 4).

| part | chosen | mass | price | what it feeds |
|---|---|---|---|---|
| arm actuators ×2 | Steadywin **GIM8108-8** with the GDS68 driver: the legs' part (6.76 N·m continuous at 48 V, 22 peak; `legs.actuator.GIM8108_8`) | 396 g each | €124.95 (OpenELAB, Munich) | `ArmSpec.motor`; the shoulder's joint and the forearm's tendon |
| link tubes | carbon-fibre tube, 20/18 mm woven roll-wrapped (Easy Composites): 0.25 m upper arm, 0.35 m forearm | 86.5 g/m (maker), 52 g for both | €23.45 a metre, ex VAT (Easy Composites EU, Sept 2026) | `arm.TUBE_R`, `ArmSpec.upper`/`fore` |
| link fittings and pivots | aluminium tube clamps and the elbow's and wrist's pivots on **6800-2RS** ball bearings (10 × 19 × 5 mm; NSK: 1.89 kN dynamic) | 5 g a bearing (NSK); the fittings `null` — no design yet | €7.60 a bearing (SKF 61800-2RS1, kugellager-shop.net, 2026-09-28) | `ArmSpec.upper_mass`/`fore_mass`, 0.12 kg each with the tubes, ESTIMATED |
| parallelogram rods ×3 | the elbow's drive rod beside the upper arm, and the level linkage's two | `null` — no design yet (a carbon or aluminium rod with rod ends) | `null` — no design yet | `ArmSpec.rods_mass`, 0.06 kg, ESTIMATED |
| end plate, fork and lean-pad | the plate on the wrist pivot; two prongs with V-notches at ±85 mm, their flanks 60° and 31 mm long (bare metal: they are the poles); 53° end-stop ramps 1.5 mm past a peg's ends; a round lean-pad bar 3 mm behind a seated tool | `null` — no design yet (machined aluminium or printed) | `null` — no design yet | `ArmSpec.plate_mass`, 0.08 kg ESTIMATED; `ForkSpec` |
| ramp faces | acetal inserts or PTFE tape on the end-stop ramps: μ 0.15 against a steel peg | — | — | `arm.RAMP_MU` |
| tool pegs | the rover's 6 mm split steel peg (Parts.md, "Tool hub & modules": two conductors on an insulating bush) lengthened to 220 mm | 29 g (the rover's 20 g for 150 mm) | as the rover's | `rack.PEG_HALF`, `rack.peg_kg` |
| the rack | a back board and a rail, each bay's two V-trays at ±45 mm on brackets hung from the rail; three bays at 0.30 m, pegs 0.50 m up | `null` — no design yet | `null` — no design yet | `rack.RackSpec`, `rack.TRAY_Y` |
| rack tags | six 60 mm tag36h11 tags (ids 29–34), a pair a bay at ±75 mm, 0.40 m up | — | — | `rack.RACK_TAG_IDS`, `rack.tag_layout` |

**Why the legs' motor, not a smaller one.** The arm's worst static load at
any target is 2.1 N·m, and flown it never passed 4.2 N·m (down a flight of
stairs), with an RMS under 1.3 on the flat — well inside either candidate. Steadywin's
GIM6010-8 with the GDS68 would do the work, but it weighs 388 g against 396
for 5.16 N·m continuous at 48 V (their GDK table; 5 N·m rated at 24 V in the
GDS table, 11 peak; which winding the store's 48 V option ships is not
stated, and the two tables list different ones). The same mass for less torque, and
the GIM8108-8 is one spare, one driver and one set of gains with the legs.

**What was not chosen, and why:**
- **A wrist motor** (option b), e.g. a Steadywin **GIM4310-10** with the
  GDK34 (227 g; 1.35 N·m continuous, 10.26 stall at 48 V; €143.95 at
  OpenELAB): 44 % more holding torque at full reach than the parallelogram,
  itself at 61 % of its rating, for a plate level against the body's pitch
  that a hung tool does not need.
- **A lock**, for now — a magnet in each V (supermagnete **S-10-03-N**,
  10 × 3 mm N42: 17.7 N on a thick steel plate, and about a third of that
  at 1 mm, read by eye off the maker's chart; €0.50) or a spring catch.
  Gravity held the tool through every walk and trot and up the stairs;
  coming down, about 1 descent in 60 floats the peg at a step (the fork
  falls faster than g) and the tool goes. The magnet is the candidate if
  step 4 finds tools carried downstairs often; in a fall a lock would keep
  the tool on an arm the robot rolls across.
- **Pogo pins at the coupling** (Mill-Max **0906-1-15-20-75-14-11-0**:
  6.5 A at a 30 °C rise, 10 g free and 60 g at 0.71 mm, $0.86 at DigiKey):
  the peg stays the connector. Its preload is the tool's weight, and flown
  it opened for at most 88 ms (down the stairs) against the rover's 200 ms
  holding capacitor.

---

## Scale — how big is any of this?

**Measured off the models** (`room_hub.xml`, arm stowed), because the numbers
are easy to misread: MJCF `size` values are **half**-extents, so the chassis
line `size="0.12 0.09 0.03"` means a **24 × 18 × 6 cm** slab, not 12 × 9 × 3.

| | Footprint | Height |
|---|---|---|
| **Robot** (over wheels, mast up) | 24 cm long × 22 cm wide | **50 cm** |
| — chassis slab alone | 24 × 18 cm | 6 cm thick |
| — track width (wheel centres) | 21 cm | wheels ⌀90 mm |
| — battery bay | 13 × 4.6 × 2.6 cm | matches a real 5000 mAh 3S pack |
| **Tool rack** | 189 cm wide × 17 cm deep | **55 cm** |

So: a robot roughly the size of a shoebox with a 50 cm mast, **2.44 kg** in
the model (fork robot with LIDAR and arm; the plug robot is 2.34 kg), and a
rack as wide as a bookshelf: five tool bays at 0.25 m pitch plus the charge
bay is 1.86 m of rail (`rack/coupling.py` `RACK_HALF_W` = 0.93, re-checked
against both rooms it stands in — ToolPattern.md §6).

**Build medium.** The rack does NOT fit a typical 220 × 220 mm print bed, and
should not try to: print the *brackets, V-trays, tag plates and fork* (all
small, all tolerance-critical), and make the **rail, posts and base from
stock material** — wood is entirely reasonable, as is 2020 aluminium
extrusion if you want the bay pitch to be adjustable later. Only the parts
that touch the peg need print accuracy.

**Electronics volume.** A Pi 5 is 85 × 56 mm; the 24 × 18 cm chassis has room
for Pi + motor driver + battery without growing. No accelerator HAT is
needed: desktop timings for the stages that transfer to hardware, scaled a
pessimistic 5× for a Cortex-A76, came to ~78 ms per perception cycle.

## Drive system

### Motors (2×) — ✅ CHOSEN: Pololu 50:1 (July 2026)

**Pololu 50:1 Metal Gearmotor 37Dx70L mm 12V with 64 CPR Encoder (Helical Pinion)** — Pololu #4753

- Source: [Eckstein-shop.de](https://eckstein-shop.de/Pololu-501-Metal-Gearmotor-37Dx70L-mm-12V-with-64CPR-EncoderHelical-Pinion-EN) — **€84,43 each**. Datasheet: [pololu.com/product/4753](https://www.pololu.com/product/4753)
- Catalog entry `gearmotor_37d_50`: the 50:1 ratio is the wheel joint's
  `armature` (reflected rotor inertia ∝ ratio²) and `damping` (the gearbox's
  ~30–35 % torque loss), both load-bearing for sim stability — SimNotes
  "Physics modeling rules"; stall torque 2.06 N·m and no-load 20.9 rad/s are
  the motor actuator's `forcerange` / `ctrlrange` and `power.STALL_TORQUE` /
  `NOLOAD_SPEED`; 5.5 A stall / 0.2 A no-load are `power.STALL_A` /
  `NOLOAD_A`. The 64 CPR encoder (3200 CPR at the output) is
  `perception/encoders.WHEEL_COUNTS_PER_REV`: the reckoner counts whole
  counts (#386). 6 mm D-shaft, which sets the wheel choice below.

Runner-up, not selected: the 30:1 sibling (#4752, same price; 330 rpm /
1.37 N·m) — passed over for the 50:1's push force, because speed is a low
priority for this robot.

### Wheels (2×)

⚠ **Finding:** Pololu's 60–70 mm wheels only fit 3 mm shafts. For the 37D's **6 mm D-shaft**, the verified Pololu path is a 90 mm (or 80 mm) wheel + universal mounting hub.

Catalog entries `wheel_90x10` (#1435–1439, €11,13 / pair, mass TBD),
`hub_6mm_m3` (#1999, the set-screw hub the wheel bolts to) and the
alternative `wheel_80x10_multihub` (whose inserts are 3/4 mm only — unverified
that it accepts the 6 mm hub).

✅ **Decided (July 2026): 90×10 mm** → wheel geom r = 0.045 m
(`control.WHEEL_RADIUS`), half-width 0.005. With the 50:1 motor, top speed
is 20.9 × 0.045 ≈ **0.94 m/s** and stall push ≈ 2.06/0.045 ≈ 46 N per wheel.
The larger radius destabilised the sim's pitch dynamics until tyre
compliance (`solref="0.05 1"`) and gearbox damping were modelled — SimNotes.

---

## Chassis & mechanical

Catalog entries `ball_caster_19mm` (Pololu #955, ≈ €4,50; ⚠ its low
friction needs `priority="1"` or MuJoCo takes the pair MAX — SimNotes "THE
caster lesson"; ⚠ the sim's sphere is 40 mm across, the caster's stance
rather than its 19 mm ball), `bracket_37d` (sets axle height; open decision
4 is blocked on it), `chassis_plate` (track width 0.21 m is set by the
bracket spacing) and `bumper_switch` (issue #94: a sprung bar over two
Omron D2F-01L2 hinge-roller-lever microswitches — sourced in #199, 0.78 N
operating force, €2.59 — 6–12 cm off the floor, feeding
`HubSwap.pressing` — dead reckoning holds its travel while the bumper is
pressed on the side the wheels roll toward. Why: a drive stalled against a
fence pumped 4.28 m of imaginary travel in 30 s, and motor torque cannot
tell a stall from a cruise. A rigid chassis bounces off a rigid fence, so the
sim holds the press `PRESS_RELEASE_S` past the last contact; a sprung bar
stays pressed through that. SimNotes "A stalled drive is an odometry pump").

---

## Vision & ranging

### ✅ CHOSEN (Aug 2026): 2D LIDAR + ONE navigation camera — stereo dropped

**Decision.** The hub-era robot ranges with a scanning LIDAR and sees with a
single camera; the stereo pair is retired. Applied to `pluggybot_fork.xml`.
`pluggybot.xml` (plug robot) keeps its pair, frozen for milestone 6–7
reproducibility.

**Why, measured.** Real SGBM on this sim's own stereo pair, in the best case
a stereo pair could ever have, produced disparity for only **49.7 %** of the
mapper's scan row at **593 mm** median error mid-room, against a 50 mm grid
cell; geometry agrees (σ_z ±649 mm at 5 m on a 60 mm baseline). Stereo cannot
build this map: the room is flat painted walls, the classic no-disparity
case. Tables and method: SimNotes "Sensor-realism pass".

Catalog entries `lidar_rplidar_c1` (Slamtec RPLIDAR C1 or A1M8: 360°,
10 Hz, 12 m, ±30 mm, ~110 g, ~2.5 W — the maker's datasheet says 1.15 W
typical, found by #377 and left for the rover's last days. `perception/lidar.py` casts 360
rays (one `mj_multiRay`) at the part's 10 Hz with ±10 mm + 1 % noise and 2 % dropout,
under-ranges it to 8 m on purpose, drops self-hits; its offset from the axle
is `LIDAR_ORIGIN`, which the grid update bakes in — move the unit, move the
constant) and `pi_camera_3` (×2: `left_eye` on the head for AprilTags,
`dock_eye` on the lift carriage so it rises with the fork; the IMX708's 41°
vertical FOV is both cameras' `fovy`).

**What it bought**
- Two cameras on the Pi 5's two CSI ports — no multiplexer (the old
  third-camera blocker, closed).
- Drops SGBM's ~60 ms/frame from the Pi 5 perception budget
  (~138 → ~78 ms per cycle).
- 360° coverage, where the forward ±20° safety reflex could never protect a
  spin (SimNotes, milestone 4); works in the dark and on textureless walls.

**What it costs**
- **~2.5 W continuous** → `power.ELECTRONICS_W` 6.0 → 8.5, straight off run
  time.
- **~110 g mounted high** (body-local z = 0.15) for a clear scan plane at
  0.223 m.
- **A ~17° blind sector behind-right** where the mast stands in the scan
  plane, centred at `atan2(−0.05, −0.10)`; self-hits are dropped rather than
  reported as free space (`tests/test_lidar.py` pins both).
- The outlet-landmark projection lost its depth source and would need a
  bearing + wall-intersection rewrite — plug-anywhere path only.

Superseded, for the record: **DIY stereo** (2× Camera Module 3 on a custom
60 mm bracket, July 2026) was the right pick against the **Luxonis OAK-D
Lite** (€193–199, 75 mm baseline, 35 cm minimum depth), but the question
neither asked was whether any stereo pair could produce the mapper's scan.

### ✅ CHOSEN (Sep 2026): near-field depth camera — RealSense D435 (issue #34)

**Decision.** A third ranging sensor, for the one place the scan plane never
looks: the floor. An **Intel/RealSense D435** (active IR stereo, 87° × 58°,
848 × 480 depth, 50 mm baseline, 72 g (75 g in the March 2026 datasheet), 90 × 25 × 25 mm, USB 3) on the **mast
top, over the axle, pitched 40° down**. Applied to `pluggybot_fork.xml`
(`depth_cam_body`, `depth_eye`); `perception/depth.py` is the sim,
`perception/heightmap.py` the robot-centric 2.5D map it feeds, and
`scripts/nearfield_spike.py` keeps every number below measured.

**The candidates, and why each lost** — decided on lessons this repo had
already paid for, not on a spec sheet:
- **RealSense D405** (7–50 cm, the true near-field unit): *passive* stereo,
  no projector. The DIY pair's measured failure (49.7 % of a scan row, on
  painted surfaces) is this camera's failure on a painted floor and on the
  matte printed modules it would be looking for. Rejected on that number.
- **A CSI time-of-flight module** (Arducam ToF class, ~€50): honest and
  cheap, but the Pi 5's two CSI ports are the two cameras — it reopens the
  third-camera blocker the LIDAR swap closed.
- **A second RPLIDAR tilted at the floor** (~€90, 110 g): one line across the
  floor, swept only while driving; it cannot look at a thing from a
  standstill, which is what grasping from a discovered pose needs (#34's
  step 4). Also 110 g more, high.
- **D435** wins because the projector makes textureless surfaces work, the
  depth is computed on the unit (no Pi 5 budget beyond USB), and it is the
  standard mast-top pick (Stretch carries one at its head).

**The mount, measured** (`--mount`; the deck lesson in SimNotes). Head height
was tried first and the frame was all chassis deck and LIDAR body. From the
mast top the centre column sees the floor from **0.26 m ahead of the axle
(6 cm past the bumper) at ANY pitch ≥ 35°** — the deck shadows nearer floor
whatever the pitch — so pitch sets the FAR edge alone: 2.1 m at 40°, 1.7 at
45°, 1.0 at 55°. 40° keeps 88 % of the frame on the floor and 3.9 % on the
robot itself. Above the LIDAR plane, so the blind sector is unchanged; 15 mm
above the carriage's top of travel.

**What it costs the body.** Mass 2.444 → 2.516 kg; CoM +10.7 mm; a
full-throttle step launch pitches 3.9° → 6.4° (the mission ramps at
30 rad/s² and never commands that), a cruise launch 0.9° → 1.3°; braking and
open-loop veer unchanged. The y-counterbalance test still holds (CoM
−6.4 → −7.7 mm, bar 10 mm).

**What the sim keeps honest** (each pinned in `tests/test_depth.py`): axial
depth with the datasheet's limits on z (`MIN_Z` 0.28 m — which does not bind
on this mount, nothing in frame but the robot is inside 0.5 m — and `MAX_Z`
3 m, under the part's 10 on purpose); noise **quadratic in z**
(0.08 px of disparity → 3.6 mm at 1 m, 14 mm at 2 m); the **occlusion
shadow** on the image-left of every near edge, `f·B·(1/z_near − 1/z_far)`
pixels wide; the robot's own deck dropped (and still occluding); and **no
return is NOT a reading** — the LIDAR's rule inverted: an out-of-range pixel
is unknown, never free space. Sim resolution is 120 × 70 (the part's aspect
at 1/7), 8400 batched ray casts at **~5–7 ms a frame** on both worlds
(60 × 35 is ~1.7 ms), deterministic and off the GPU. Its draw is
`power.DEPTH_CAMERA_W` (2.0 W: the community-measured ~1.9 W streaming with
the projector; the datasheet figure is open decision 10), on the pack only
while the map is being built — `HubLifecycle(near_field=)`, on when served
and off in a test — and `economy/energy.json` is measured with it on.

**The representation, measured** (`--cost`; the issue's own question). A
robot-centric **2.5D height map**, 4 m square at 2 cm = **40 000 cells**
(the 2D grid is 56 000 at 5 cm), world-axis aligned and recentred by whole
cells, the highest point per cell from the LAST frame to see it: **0.7 ms an
update**, 480 KB. A voxel map of the same window to 0.5 m is 250 000 cells at
2 cm (2 M at 1 cm) and, with the free-space carving that lets it forget a
moved object, **111 ms an update** as written in the spike (~25 ms even at
the 2D grid's per-sample efficiency) — 25× the cells and 40–200× the time to
answer a question the height map answers. Voxels earn their keep only when
what is UNDER an overhang matters; today that is nothing.

**What it finds from a standstill** (`--find`): a 5 cm cube at 0.4–1.6 m and
a 10 cm one to 1.2 m, height to 3 mm; a 3 cm cube only inside ~0.8 m, a 2 cm
one never — the rows land ~3 cm apart on the floor at 1 m, so the map
integrates over motion and `HeightMap.things` bridges one unmeasured cell.


---

## Arm (milestone 6)

### Naming

**Mast** = the fixed vertical column. **Lift** = the carriage that travels up
it (and the actuator driving that). **Telescoping arm** = the horizontal
extension carrying the fork. Hello Robot Stretch's
vocabulary and architecture: base owns x/yaw, lift owns z, arm owns reach.

### ⚠ The mass budget is the binding constraint

Measured in sim (headless force probe, ramped loads on the armed robot at
0.30 m): pushing forward holds **~3 N then slides**; backward holds ~4 N
then goes caster-light. So the arm's working forces stay small — a
compliant wrist, and forward docking (mapping and cameras face forward; the
failure mode is a benign slide) — and a tool that must press gets the
lean-pad (ToolPattern.md, "the force budget").

What came out of it and still binds: **battery position is a design
variable, not packaging** — x = +0.05 (ahead of centre) for tipping margin,
**y = +0.06 as a counterweight** for the arm assembly hanging at y = −0.05,
without which the robot veers 26 cm right over 4 m open-loop (the catalog's
`battery.pos` feed pins the position); carrying the heaviest module adds
5.5 mm over 2.7 m and needs no re-tune.

### Lift and telescoping arm — 2× linear actuator

**igus drylin® E lead-screw stepper linear actuator, NEMA11** — igus DLE-LA-0001

- Source: [igus.com/product/DLE-LA-0001](https://www.igus.com/product/DLE-LA-0001). Price: **TBD** — igus quotes stroke-configured units through their configurator, not a fixed list price. Get a quote for both axes together.
- Catalog entry `igus_dle_la_0001`: 50 N thrust is both axes' `forcerange`
  (6× the plug era's worst-case 7.8 N insertion); the 0.12 N·m holding torque
  holds position unpowered, which is why `power.ACTUATOR_W` is drawn only while
  moving and a parked module axis is a position servo at its target; the
  dryspin® 6.35 × 5.08 screw feeds 0.0254 mm per 1.8° step, far finer than
  the ±3 mm docking budget. Stroke is configurable — want ~0.25 m lift and
  ~0.20 m reach (the model's lift travels 0.31 m). Mass TBD, and it matters
  (mass budget above).

Reach is set by parking the base at ~0.25 m; a 0.6 m cantilever is
unaffordable on this chassis.

### Compliant wrist (passive)

A **remote center compliance (RCC)**: a sprung mount between the arm tip and
the tool that lets it translate and rotate slightly, so contact forces from a
small misalignment *correct* the error instead of jamming. Published RCC
devices recover up to ~4 mm and ~9° — more than the ±3 mm / ±3° budget, with
no control loop. Build, don't buy: four compression springs plus a floating
plate, **≈ €10**. The sim models it as 150 N/m lateral and 1 N·m/rad angular
(`coupling.LAT_STIFFNESS` / `YAW_STIFFNESS`, and the fork model's wrist
joints); those were guesses — measure the built part and update, since the
whole tolerance envelope scales with them.

---

## Tool hub & modules (milestone 8)

The coupling spike (`scripts/hub_spike.py`) validated the fork-and-peg
gravity latch: ±4 mm lateral / <2° yaw envelope, retention beyond the base's
traction limit, 300 g tool mass with margin (ToolPattern.md §2 is the
envelope). Parts below are the build list that geometry implies; **prices
and models TBD.**

| Part | Route | Notes → sim |
|---|---|---|
| Rack: rail, posts, shelf, V-trays, wall braces | rail/posts/base from stock (above); trays and brackets **3D-printed** (PETG; the trays see ~3 N loads) | geometry = `rack/coupling.py` constants; 1.86 m of rail for five bays + charge bay |
| Bay presence switches | one Omron D2F-01L2 per bay (eight, €2.59 each) in the +y V-tray; a board on the rack reads them and reports them over the network (`rack_controller`, none chosen) | `coupling.bay_switches`: which bays are taken, never by which module — what the robot's `rack` context is built off (Overseer.md §2i). The sim reads contact, not force: a bare switch under one tray would not close for the LCD or the plug (0.70 and 0.77 N a tray against its 0.78 N), so the lever has to carry the tray, or a lighter switch; open |
| Tool peg axles | **6 mm steel rod** (conductive — see below), 2× 63 mm conductors on a 24 mm insulating centre bush, 150 mm overall | the one loaded part **and the electrical connector** — `PEG_R`, `PEG_HALF`, `PEG_INSUL_HALF` / `PEG_COND_HALF` |
| Arm fork + V-notches | 3D-printed, mounts where the plug's RCC sits | prong stance ±58 mm (`FORK_Y`) |
| Module frames (LCD, plug, pen, claw, seed dispenser) | 3D-printed plates, common peg interface | 130–211 g measured, ~250 g practical ceiling (ToolPattern.md "Mass and geometry class") |
| Module electronics | 1× ESP32-class board per module (~€5 each) | **power-only coupling, wireless data** — keeps the mating interface dumb and tolerant. Modelled as a 0.6 W load (`power.MODULE_IDLE_W`) drawn only while the coupling conducts |
| **Module power contacts** | **none to buy — the peg IS the connector** | Split peg + the fork's two V-notch pairs = a two-pole coupling with 0.43–0.47 N of gravity preload per plate, already there, self-wiping on the seating slide. Needs a conductive rod, an insulating centre bush, isolated V-plates, and a **holding capacitor sized for ~200 ms** — measured worst outage 178 ms under hard driving with honest peg friction (SimNotes "The module electrical interface"; ToolPattern.md §3) **Rated 12 W at 12 V, 1 A** (`coupling.PEG_POWER_W`, a design decision raised from 6 W on 2026-09-15): a light steel point contact is tens of milliohms to ~0.1 Ω, so 1 A dissipates ~0.1 W there and the limit on plain steel is drop and fretting, not heat — a paper number until the built coupling is measured. Open to the hardware upgrade if a build ever wants more, in the order that keeps the peg as the connector: brass sleeves on the conductor segments and plated V-plates, then a sprung contact in the V (the charge contacts' pogo trick); a separate pogo pair on the plate, or 24 V, would give up the tolerance this design was chosen for |
| Charge contacts | pogo-pin pairs (spring-loaded, ~€5) on the hub face at bumper height (`CHARGE_PIN_Z` 0.09), pads on the robot | preload from the drive-in press; the electrical-contact criterion carries over verbatim |
| Hub power | 12.6 V CC/CV charger board (3S, ~€10–15) fed by a mains adapter; balance leads handled robot-side by a 3S BMS | replaces wall-outlet charging as the primary path |
| LCD module | small SPI/I2C display driven by the module's ESP32 | display-only; the face is drawn in the browser off a streamed enum |

Open for the physical design: pogo-pin placement that engages by the same
drive-in motion (no extra alignment), and whether the trays need steel wear
inserts.

### What the workshop may build from (issue #199)

The agent's tools (`workshop/`, Overseer.md §2d) are built from the
catalog shelf of `protocol/parts.json`, and a part is buildable-from only
when the catalog knows every number the validator reads — mass, size, the
motion's numbers, the draw. Sourced one at a time, each off the maker's
datasheet, never a seller's summary:

| Part | Chosen | The numbers that made it usable |
|---|---|---|
| micro servo | FEETECH FS90-FB (#168) | 0.147 N·m, 800 mA stall at 6 V = 4.8 W |
| second servo, metal gear | FEETECH FS90MG | 0.216 N·m at 6 V at the SAME 800 mA stall, 12.7 g. Passed over: Tower Pro MG90S (no published current), Power HD HD-1810MG (1.4 A stall = 8.4 W, over the peg alone) |
| slide | Actuonix L12-100-50-6-R | 100 mm stroke, 22 N lifted, 25 mm/s, 460 mA on the 6 V winding = 2.76 W, 56 g, €77. ⚠ The L16-140 that would cover the pen's 110 mm travel publishes its stall current at 12 V only, so its 6 V draw would be a guess; the pen's carriage stays the unspecified `module_lead_screw`, off the catalog shelf |
| microswitch | Omron D2F-01L2 | 0.78 N OF, 12.8 × 5.8 body, 16.5 mm free position, ≈0.5 g (the datasheet's pin-plunger figure; the lever's fraction of a gram is unpublished). On a tool it is a `contact` sense |
| eye | Ai-Thinker ESP32-CAM | the camera whose maker publishes a draw — 0.9 W flash off, 1.55 W flash on — and the one that fits "power-only coupling, wireless data", because it is the radio. Raspberry Pi still publishes none for Camera Module 3 (both product briefs, the forum, Arducam's and InnoMaker's IMX708 sheets checked, 2026-09-14), so `pi_camera_3` stays unbuildable-from |

⚠ **The peg's budget was the binding constraint at 6 W** (`coupling.
PEG_POWER_W`): one servo at stall (4.8) plus the eye with its flash (1.55)
plus the module's 0.6 W is 6.95 W, so the first tool that looked and moved
was refused — the re-examination #199's acceptance deferred to "the first
tool that wants two actuators", one part early. Raised to 12 W (1 A at
12 V) on 2026-09-15 with the argument in the constant's comment and the
"Module power contacts" row above: a servo and an eye fit, two servos and
an eye fit (11.75 W), three servos at stall (15 W) do not. The validator
still sums every part's ceiling as if simultaneous.

## Power

### Battery — 3S LiPo (recommended), part TBD

The motors are 12 V; a 3S LiPo is 11.1 V nominal / 12.6 V charged, which suits
them directly. 4S (14.8 V) would need a buck converter. LiFePO4 4S (12.8 V) is
the safer, heavier, longer-lived alternative and is worth pricing too.

| Spec | Target | Why / → sim |
|---|---|---|
| Chemistry / cells | 3S LiPo (11.1 V) | matches the 12 V gearmotors without regulation → `power.NOMINAL_V` |
| Capacity | **≥ 5000 mAh** (~55 Wh) | 2× 5.5 A stall motors + Pi 5 (~5 A @ 5 V through a buck). `power.CHARGE_W` 55 W is ~1C into it; capacity in the sim is a knob (demo and hosting cells), `--battery-wh 55.5` runs the real pack |
| Continuous discharge | ≥ 20 C | headroom over the ~11 A both-motors-stalled worst case |
| Connector | XT60 | standard, and matches the Cytron MDD10A wiring |
| **Mass** | **~400 g** | **a feature, not a cost — it is the traction ballast** (mass budget above) |
| Mount position | **x = +0.05, y = +0.06, as low as possible** | traction 7.1 → 8.2 N, tipping 2.5 → 4.2 N; the y offset is the counterweight |

Widely available (Gens Ace, Ovonic and similar in this size); **TBD: pick a
specific pack from a German retailer and confirm physical dimensions against
the chassis plate.** Also needed: a 3S balance charger, and a **12 V → 5 V
5 A buck converter** for the Pi 5.

⚠ The models already carry the pack as a 400 g placeholder box at that
position. A real pack of a different mass or footprint moves every physics
threshold derived from the model, which is why mass is budgeted last.

## Electronics — later (low priority)

- **Motor driver:** [Cytron MDD10A](https://botland.store/drivers-for-dc-motors/15818-cytron-mdd10a-dual-channel-30v-10a-motor-controller-5904422350444.html) (Botland, price TBD ~€20-class) — dual channel, 10 A continuous / 30 A peak per channel at 5–30 V, comfortably covers the 5.5 A stall current per motor.
- **Compute:** Raspberry Pi 5 8 GB — [BerryBase](https://www.berrybase.de/en/raspberry-pi-5-8gb-ram); ⚠ cheapest Geizhals listing July 2026 is **€202,90 incl. active-cooler kit** ([Geizhals](https://geizhals.de/raspberry-pi-5-modell-b-a3096144.html)) — well above the historical ~€90 board price (RAM price surge); verify standalone-board price before budgeting.

---

## Open decisions

1. ~~Stereo camera~~ → **Superseded (Aug 2026): 2D LIDAR + one camera** — "Vision & ranging".
2. ~~Wheel diameter~~ → **Decided (July 2026): 90×10 mm** (sim r = 0.045).
3. ~~Gear ratio~~ → **Decided (July 2026): 50:1 (#4753)** — push force over top speed.
4. Chassis material/supplier (Misumi/igus vs laser-cut) — blocked on motor-bracket choice.
5. ~~Dock forward or backward?~~ → **Decided (Aug 2026): forward** — a benign ~3 N slide, and the cameras face that way ("The mass budget").
6. **Battery pack** — specific 3S LiPo (or 4S LiFePO4) from a German retailer, with
   dimensions checked against the chassis plate. Position x = +0.05, y = +0.06, low.
7. ~~Third camera routing~~ → **Closed (Aug 2026) by the LIDAR swap**: two cameras on two CSI ports, LIDAR on USB/UART.
8. ~~Plug body diameter~~ → **Moot**: the plug era is retired (#376).
9. **Lift/arm stroke + price** — get an igus quote for two NEMA11 lead-screw actuators
   (~0.25 m and ~0.20 m stroke) and their masses.
10. **Depth camera sourcing and draw** — RealSense left Intel in 2025: confirm
    an EU distributor and price for the D435 (roughly €300–400, unverified),
    and its streaming draw against `power.DEPTH_CAMERA_W`'s 2.0 W.
