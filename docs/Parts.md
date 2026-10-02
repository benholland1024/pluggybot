# Parts List

Specs and sourcing for the real parts the sim is modelled on, and the sim
parameter each number feeds. Hardware honesty (PluggyPlan.md, "What stays
fixed") is why this file exists: every part is purchasable, and a sim
constant with no part behind it is a guess and is marked as one.

> **The list is data:** `protocol/parts.json`, emitted by `uv run python -m
> pluggybot.rack.catalog` (issue #185) — every part below with its number,
> source, mass, price and the sim constant it feeds, the constant's value
> READ OFF THE MODEL when the file is built, so the fixture cannot describe a
> number the sim does not use. A number this doc does not know is `null`
> there with a reason, never a guess. Two shelves: `build`, the bill of
> materials below, and `catalog`, what the robot's workshop may build a tool
> from ("What the workshop may build from"). The website's parts page
> renders it. This doc keeps the reasoning; the spec→parameter tables live
> there.

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
| [RealSense D435 depth camera (active IR stereo)](https://www.digikey.de/de/products/detail/realsense/82635AWGDVKPRQ/9926002) `D435` | 1 | 340.01 | 340.01 | none in stock, the maker's standard lead 14 weeks (DigiKey DE); 21 days at MyBotShop for €419.99 | `depth_eye.fovy`, `legs.model.ELECTRONICS_W['depth camera']`, `legs.model.MassBudget.depth_cam`, +5 |
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
| **tools** | | | **163.95** | | |
| Tool peg, 220 mm: two 6 mm steel conductors on an insulating bush | 4 | allowance | 30.00 | unknown: nothing drawn to quote | `legs.rack.PEG_HALF`, `legs.rack.PEG_MASS` |
| [FEETECH FS90MG micro servo (digital, metal gears)](https://eckstein-shop.de/Feetech-FS90MG-6V-22kgcm-Digital-Servo) `FS90MG` | 2 | 5.95 | 11.90 | unknown: not read for this bill; the price is the workshop catalog's (#199, 2026-09-14) | — |
| [Actuonix L12-100-50-6-R micro linear servo (100 mm stroke, 50:1, 6 V, RC input)](https://www.digikey.de/de/products/detail/actuonix-motion-devices-inc/L12-100-50-6-R/11689540) `L12-100-50-6-R` | 1 | 77.05 | 77.05 | unknown: not read for this bill; the price is the workshop catalog's (#199, 2026-09-14) | `legs.rack.PEN_TRAVEL`, `legs.rack.PEN_FORCE_N`, `legs.rack.PEN_SPEED` |
| Linear Hall-effect sensor and a small magnet on the pen's sprung quill, reading its travel | 1 | allowance | 5.00 | unknown: no sensor chosen | `tools.drawing.QUILL_TOUCH` |
| ESP32-class board, one per module | 4 | 5.00 | 20.00 | unknown: no board chosen | `power.MODULE_IDLE_W` |
| Small SPI/I2C display driven by the module's ESP32 | 1 | allowance | 20.00 | unknown: no display chosen | `legs.rack.SCREEN_HALF` |
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
| allowances (nothing designed yet) | 1,575.00 | |
| contingency, 15% | 1,132.93 | |
| **total** | **8,685.82** | **9,904.44** |
| budget | 17,539.24 | 20,000.00 |

**The allowances**, each a sum set aside, never a price:
- **The pack's printed case, fish paper, heat-shrink and 12S balance lead** (power), €30.00: the case printed in the shop's PETG; fish paper, heat-shrink and a 12S balance lead
- **The torso's frame: two 3 mm aluminium side plates, end caps and cross members** (frame and legs), €300.00: six to eight 3 mm AlMg3 parts: AluFritze's online configurator prices a plain 300 x 150 mm rectangle at €20.02 (four for €80.10, 10-14 working days); a plate with holes is priced from its drawing
- **The legs' motor mounts, hip brackets, knee housings and tube clamps, with their fasteners** (frame and legs), €800.00: twelve motor mounts, four hip brackets and eight tube clamps, nothing drawn, sized as machined aluminium at a job shop; printed on the shop's P1S they cost their filament
- **The arm's end plate, fork and lean-pad (two prongs, four 60° V's, two 53° end-ramps), and the parallelogram's three rods** (arm), €150.00: one machined aluminium plate with its prongs, V's and ramps, and three 5 mm rods threaded M5: a plain 4 mm plate is €23.12 at AluFritze, the V's and ramps are machining
- **Tool peg, 220 mm: two 6 mm steel conductors on an insulating bush** (tools), €30.00: four pegs cut and turned from a metre of 6 mm silver steel and an acetal bush each, as `peg_rod_6mm`
- **Linear Hall-effect sensor and a small magnet on the pen's sprung quill, reading its travel** (tools), €5.00: a linear Hall sensor and a 3 mm magnet on the pen's quill, as TI's DRV5055: a few euros; none chosen yet
- **Small SPI/I2C display driven by the module's ESP32** (tools), €20.00: a small SPI display for the LCD tool's ESP32; none chosen yet
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
- **Open in #379:** the computer's budget (the bill carries a Pi 5 8 GB;
  the 16 GB is in stock), the tools' supply voltage at the peg, and every
  drawing an allowance waits on.

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
| 2D LIDAR | Slamtec **RPLIDAR C1** ("Vision & ranging", below) | 110 g | €84.90 (Funduino, in the bill) | 1.15 W, the maker's 230 mA at 5 V |
| depth camera | RealSense **D435** ("The near-field depth camera", below) | 75 g (datasheet, March 2026) | €340.01 (DigiKey DE, in the bill) | 2.0 W streaming depth with its projector |
| compute, camera, IMU | Raspberry Pi 5 8 GB, a Camera Module 3, the ICM-42688-P | in `electronics` | in the bill | 6.0 W, an estimate: Raspberry Pi publishes no load figure |
| frame | aluminium side plates and cross members | 1.0 kg (estimated) | `null` — no design yet | `MassBudget.frame` |
| thigh, shank, foot | tube, no belt (the knee is direct), rubber foot | 0.21 kg a leg (estimated) | `null` — no design yet | `BodySpec.*_mass` |
| #378's arm | chosen by #378 ("The arm and its coupling", below); the served body's since #405 (`BodySpec.arm`; the placeholder is `model.SIZING`, what #377's tables flew) | 1.17 kg (the placeholder budgeted 0.9) + a tool up to 0.40 kg | — | the legs' torques re-flown with it and a tool aboard: the knee's p99.5 ≤ 9.4 N·m on the flat, 14.0 up the house's flight (13.5 without it); its two drivers stand by with the legs' twelve (`model.ELECTRONICS_W`) |

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

### Vision & ranging

**A 2D LIDAR and one navigation camera, never a stereo pair** (Aug 2026).
Real SGBM on the sim's own stereo pair, in the best case a pair could ever
have, produced disparity for only **49.7 %** of the mapper's scan row at
**593 mm** median error mid-room, against a 50 mm grid cell (σ_z ±649 mm at
5 m on a 60 mm baseline): rooms are flat painted walls, the classic
no-disparity case (SimNotes "Sensor-realism pass"). The LIDAR is
`perception/lidar.py`: 360 rays (one `mj_multiRay`) at the part's 10 Hz,
±10 mm + 1 % noise and 2 % dropout, under-ranged to 8 m on purpose, and
self-hits dropped rather than reported as free space. The camera reads the
tags: the rack's, the dock's and the lab's signs. It works in the dark and
on textureless walls, and 360° covers a spin, which a forward camera's
safety reflex never could.

### The near-field depth camera: RealSense D435 (issue #34)

For the one place a scan plane never looks: the floor, and what is on it.
An active IR stereo unit: 87° × 58°, 848 × 480 depth, a 50 mm baseline,
72 g (75 g in the March 2026 datasheet), 90 × 25 × 25 mm, USB 3.

**The candidates, and why each lost**, decided on lessons this repo had
already paid for, not on a spec sheet:
- **RealSense D405** (7–50 cm, the true near-field unit): *passive* stereo,
  no projector. The stereo pair's measured failure above is this camera's
  failure on a painted floor and on the matte printed tools it would be
  looking for.
- **A second RPLIDAR tilted at the floor** (~€90, 110 g): one line across
  the floor, swept only while moving; it cannot look at a thing from a
  standstill.
- **A CSI time-of-flight module** (Arducam ToF class, ~€50) lost when two
  cameras held the Pi 5's two CSI ports. The bill buys one camera, so a
  port is free and that reason is gone; the D435 stays on the other two.
- **D435** wins because the projector makes textureless surfaces work and
  the depth is computed on the unit (no Pi 5 budget beyond USB).

**What the sim keeps honest** (each pinned in `tests/test_depth.py`): axial
depth with the datasheet's limits on z (`MIN_Z` 0.28 m, and `MAX_Z` 3 m,
under the part's 10 on purpose); noise **quadratic in z** (0.08 px of
disparity → 3.6 mm at 1 m, 14 mm at 2 m); the **occlusion shadow** on the
image-left of every near edge, `f·B·(1/z_near − 1/z_far)` pixels wide; the
robot's own body dropped (and still occluding); and **no return is NOT a
reading** — the LIDAR's rule inverted: an out-of-range pixel is unknown,
never free space. The frame is 120 × 70 (the part's aspect at 1/7), 8400
batched ray casts, deterministic and off the GPU.

**The representation, measured.** A robot-centric **2.5D height map**
(`perception/heightmap.py`), 4 m square at 2 cm = 40 000 cells, updated in
0.7 ms (480 KB); a voxel map of the same window to 0.5 m, with the
free-space carving that lets it forget a moved object, took 111 ms. Voxels
earn their keep only when what is UNDER an overhang matters. The
quadruped's planner reads the camera's points, never the map (CLAUDE.md).

### The IMU and the encoders, as the robot reads them (#386)

The robot is never told the sim's exact angles or rates: `perception/imu.py`
and `perception/encoders.py` give every reading the part's own error, each
number from its sheet or a stated range.

| part | what the sim adds | from |
|---|---|---|
| IMU: TDK **ICM-42688-P** | gyro 0.0028 °/s/√Hz white noise; accelerometer 65 µg/√Hz (x, y), 70 (z) | DS-000347 rev 1.9, Tables 1–2 (tested in production) |
| | the offset left after the boot calibration: ±0.005 °/s/°C (gyro) and ±0.15 mg/°C (accel) over `TEMP_SWING_C` = 10 °C — ±0.05 °/s and ±1.5 mg, drawn per axis | the sheet's variation over temperature (characterised); the 10 °C swing is a stated choice. The initial ±0.5 °/s and ±20 mg are what the calibration removes |
| | scale ±0.5 % per axis, gyro and accelerometer | the sheet's initial tolerance (tested in production) |
| | not modelled: cross-axis (±1.25 % gyro, ±1 % accel), nonlinearity (±0.1 %) | small against rates that average out on a near-level body |
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
quadruped's arm, its coupling and the rack". Nothing is ordered (#379); the
served body has carried the arm, and the house the rack, since #405.

| part | chosen | mass | price | what it feeds |
|---|---|---|---|---|
| arm actuators ×2 | Steadywin **GIM8108-8** with the GDS68 driver: the legs' part (6.76 N·m continuous at 48 V, 22 peak; `legs.actuator.GIM8108_8`) | 396 g each | €124.95 (OpenELAB, Munich) | `ArmSpec.motor`; the shoulder's joint and the forearm's tendon |
| link tubes | carbon-fibre tube, 20/18 mm woven roll-wrapped (Easy Composites): 0.25 m upper arm, 0.35 m forearm | 86.5 g/m (maker), 52 g for both | €23.45 a metre, ex VAT (Easy Composites EU, Sept 2026) | `arm.TUBE_R`, `ArmSpec.upper`/`fore` |
| link fittings and pivots | aluminium tube clamps and the elbow's and wrist's pivots on **6800-2RS** ball bearings (10 × 19 × 5 mm; NSK: 1.89 kN dynamic) | 5 g a bearing (NSK); the fittings `null` — no design yet | €7.60 a bearing (SKF 61800-2RS1, kugellager-shop.net, 2026-09-28) | `ArmSpec.upper_mass`/`fore_mass`, 0.12 kg each with the tubes, ESTIMATED |
| parallelogram rods ×3 | the elbow's drive rod beside the upper arm, and the level linkage's two | `null` — no design yet (a carbon or aluminium rod with rod ends) | `null` — no design yet | `ArmSpec.rods_mass`, 0.06 kg, ESTIMATED |
| end plate, fork and lean-pad | the plate on the wrist pivot; two prongs with V-notches at ±85 mm, their flanks 60° and 31 mm long (bare metal: they are the poles); 53° end-stop ramps 1.5 mm past a peg's ends; a round lean-pad bar 3 mm behind a seated tool | `null` — no design yet (machined aluminium or printed) | `null` — no design yet | `ArmSpec.plate_mass`, 0.08 kg ESTIMATED; `ForkSpec` |
| ramp faces | acetal inserts or PTFE tape on the end-stop ramps: μ 0.15 against a steel peg | — | — | `arm.RAMP_MU` |
| tool pegs | the coupling's 6 mm split steel peg ("The tool coupling", below: two conductors on an insulating bush) lengthened to 220 mm | 29 g (the 150 mm peg's 20 g, at the same grams a millimetre) | an allowance in the bill | `rack.PEG_HALF`, `rack.peg_kg` |
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
  it opened for at most 88 ms (down the stairs) against a module's 200 ms
  holding capacitor ("Module power contacts", below).

---

## The tool coupling (milestone 8)

Every tool hangs by a split PEG that a fork of V-notches takes
(`rack/coupling.py`; ToolPattern.md §2 is the envelope): the rack's V-trays
catch the peg near its ends and the fork grabs it outboard of them, so
gravity is the latch and the V's depth the retention. The served rack's peg
is 220 mm (`quad_tool_peg`, "The arm and its coupling", above); the
workshop's rig (`coupling.scene_xml`) hangs a tool by the 150 mm original
(`peg_rod_6mm`).

| Part | Route | Notes → sim |
|---|---|---|
| Tool peg axles | **6 mm steel rod** (conductive, below), two conductors on an insulating centre bush | the one loaded part **and the electrical connector** — `coupling.PEG_R`, `PEG_INSUL_HALF`; `legs.rack.PEG_HALF` on the served rack |
| Module frames | 3D-printed plates, a common peg interface | ~250 g practical ceiling (ToolPattern.md "Mass and geometry class"); the served tools are 151–219 g (`legs.rack.TOOL_KG`) |
| Module electronics | one ESP32-class board per module (~€5 each) | **power-only coupling, wireless data** — keeps the mating interface dumb and tolerant. A 0.6 W load (`power.MODULE_IDLE_W`) drawn only while the coupling conducts |
| Bay presence switches | one Omron D2F-01L2 per bay (three, €2.59 each) in the +y V-tray; an ESP32 on the rack reads them and reports them over the network | `coupling.bay_switches`: which bays are taken, never by which tool — what the robot's `rack` context is built off (Overseer.md §2i). The sim reads contact, not force: a bare switch under one tray would not close for the LCD tool (about 0.74 N a tray against its 0.78 N), so the lever has to carry the tray, or a lighter switch; open |
| LCD module | a small SPI/I2C display driven by the module's ESP32 | display-only; the face is drawn in the browser off a streamed enum |
| Pen module (#406) | the pen's sideways carriage is the bill's Actuonix L12-100 (below), its whole 100 mm stroke, under the plate; a sprung quill (60 N/m, 20 mm) holding the pen, and a linear Hall sensor and magnet on the quill reading its travel (`pen_quill_hall`, none chosen: an allowance) | `legs.rack.PEN_TRAVEL` (±50 mm), `PEN_FORCE_N`, `PEN_SPEED`, `PEN_QUILL_STIFFNESS`; the quill's sense is `tools.drawing.PenPlotter.quill` -- what the plotter finds a board's face by, with 0.05 mm of noise assumed (`QUILL_NOISE`) until a sensor is chosen. The module balances on its peg (SimNotes, "Drawing on legs") |

### Module power contacts

**None to buy — the peg IS the connector.** The split peg and the fork's two
V-notch pairs are a two-pole coupling, gravity-preloaded and self-wiping on
the seating slide. It needs a conductive rod, an insulating centre bush,
isolated V-plates, and a **holding capacitor sized for ~200 ms**: the worst
outage measured under hard driving with honest peg friction was 178 ms
(SimNotes "The module electrical interface"; ToolPattern.md §3), and 88 ms
on legs, down the stairs.

**Rated 12 W: 1 A at 12 V** (`coupling.PEG_POWER_W`, a design decision): a
light steel point contact is tens of milliohms to ~0.1 Ω, so 1 A dissipates
~0.1 W there and the limit on plain steel is drop and fretting, not heat.
A paper number until the built coupling is measured, and the tools' supply
voltage at the peg is open in #379. If a build ever wants more, the
upgrades that keep the peg as the connector come first: brass sleeves on
the conductor segments and plated V-plates, then a sprung contact in the V;
a separate pogo pair on the plate would give up the tolerance this design
was chosen for.

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
| slide | Actuonix L12-100-50-6-R | 100 mm stroke, 22 N lifted, 25 mm/s, 460 mA on the 6 V winding = 2.76 W, 56 g, €77. The pen's carriage on legs is this slide's whole stroke (#406: the rover's sim gave its pen 110 mm, and a two-digit answer, 81 mm wide, fits in 100). Passed over: the L16-140, whose stall current is published at 12 V only |
| microswitch | Omron D2F-01L2 | 0.78 N OF, 12.8 × 5.8 body, 16.5 mm free position, ≈0.5 g (the datasheet's pin-plunger figure; the lever's fraction of a gram is unpublished). On a tool it is a `contact` sense |
| eye | Ai-Thinker ESP32-CAM | the camera whose maker publishes a draw — 0.9 W flash off, 1.55 W flash on — and the one that fits "power-only coupling, wireless data", because it is the radio. Raspberry Pi still publishes none for Camera Module 3 (both product briefs, the forum, Arducam's and InnoMaker's IMX708 sheets checked, 2026-09-14), so `pi_camera_3` stays unbuildable-from |

⚠ **The peg's 12 W is the budget that binds a tool.** The validator sums
every part's ceiling (a servo's stall, the eye's flash) as if simultaneous,
plus the module's 0.6 W: a servo and an eye fit, two servos and an eye fit
(11.75 W), three servos at stall (15 W) do not.

---

## Open decisions

- **The D435's streaming draw.** The sim carries 2.0 W
  (`legs.model.ELECTRONICS_W`; `power.DEPTH_CAMERA_W`), the
  community-measured ~1.9 W streaming depth with the projector, against the
  datasheet's 3.40 W for depth and 1080p colour at once.
- The build's own are #379's: "Watch before ordering", above.
