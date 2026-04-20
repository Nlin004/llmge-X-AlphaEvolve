import argparse
import os
from pathlib import Path as p
from os.path import join as pj

def create_save_dir(save_root: str) -> str:
    save_root_path = p(save_root)
    save_root_path.mkdir(exist_ok=True, parents=True)
    n = [int(d.name[3:]) for d in save_root_path.iterdir() if d.is_dir() and d.name.startswith("exp") and d.name[3:].isdigit()]
    return pj(save_root, f"exp{sorted(n)[-1] + 1}" if n else "exp1")

# --OPTION--
def generate_seed_verilog() -> str:
    """
    GENERATION 0: ISCAS-85 c880 8-bit ALU — direct assign-statement translation.
    Functionally correct, but verbose (one assign per gate node).
    The LLM will mutate this block to reduce gate count and critical-path depth.
    Interface is fixed: 60 inputs, 26 outputs, module name c880_impl.
    """
    return """
module c880_impl (
    input  N1, N8, N13, N17, N26, N29, N36, N42, N51, N55,
           N59, N68, N72, N73, N74, N75, N80, N85, N86, N87,
           N88, N89, N90, N91, N96, N101, N106, N111, N116, N121,
           N126, N130, N135, N138, N143, N146, N149, N152, N153, N156,
           N159, N165, N171, N177, N183, N189, N195, N201, N207, N210,
           N219, N228, N237, N246, N255, N259, N260, N261, N267, N268,
    output N388, N389, N390, N391, N418, N419, N420, N421, N422, N423,
           N446, N447, N448, N449, N450, N767, N768, N850, N863, N864,
           N865, N866, N874, N878, N879, N880
);

    wire N269, N270, N273, N276, N279, N280, N284, N285, N286, N287,
         N290, N291, N292, N293, N294, N295, N296, N297, N298, N301,
         N302, N303, N304, N305, N306, N307, N308, N309, N310, N316,
         N317, N318, N319, N322, N323, N324, N325, N326, N327, N328,
         N329, N330, N331, N332, N333, N334, N335, N336, N337, N338,
         N339, N340, N341, N342, N343, N344, N345, N346, N347, N348,
         N349, N350, N351, N352, N353, N354, N355, N356, N357, N360,
         N363, N366, N369, N375, N376, N379, N382, N385, N392, N393,
         N399, N400, N401, N402, N403, N404, N405, N406, N407, N408,
         N409, N410, N411, N412, N413, N414, N415, N416, N417, N424,
         N425, N426, N427, N432, N437, N442, N443, N444, N445, N451,
         N460, N463, N466, N475, N476, N477, N478, N479, N480, N481,
         N482, N483, N488, N489, N490, N491, N492, N495, N498, N499,
         N500, N501, N502, N503, N504, N505, N506, N507, N508, N509,
         N510, N511, N512, N513, N514, N515, N516, N517, N518, N519,
         N520, N521, N522, N523, N524, N525, N526, N527, N528, N529,
         N530, N533, N536, N537, N538, N539, N540, N541, N542, N543,
         N544, N547, N550, N551, N552, N553, N557, N561, N565, N569,
         N573, N577, N581, N585, N586, N587, N588, N589, N590, N593,
         N596, N597, N600, N605, N606, N609, N615, N616, N619, N624,
         N625, N628, N631, N632, N635, N640, N641, N644, N650, N651,
         N654, N659, N660, N661, N662, N665, N669, N670, N673, N677,
         N678, N682, N686, N687, N692, N696, N697, N700, N704, N705,
         N708, N712, N713, N717, N721, N722, N727, N731, N732, N733,
         N734, N735, N736, N737, N738, N739, N740, N741, N742, N743,
         N744, N745, N746, N747, N748, N749, N750, N751, N752, N753,
         N754, N755, N756, N757, N758, N759, N760, N761, N762, N763,
         N764, N765, N766, N769, N770, N771, N772, N773, N777, N778,
         N781, N782, N785, N786, N787, N788, N789, N790, N791, N792,
         N793, N794, N795, N796, N802, N803, N804, N805, N806, N807,
         N808, N809, N810, N811, N812, N813, N814, N815, N819, N822,
         N825, N826, N827, N828, N829, N830, N831, N832, N833, N834,
         N835, N836, N837, N838, N839, N840, N841, N842, N843, N844,
         N845, N846, N847, N848, N849, N851, N852, N853, N854, N855,
         N856, N857, N858, N859, N860, N861, N862, N867, N868, N869,
         N870, N871, N872, N873, N875, N876, N877;

    // Gate-level assign translation (one assign per ISCAS-85 gate node)
    assign N269 = ~(N1 & N8 & N13 & N17);
    assign N270 = ~(N1 & N26 & N13 & N17);
    assign N273 = N29 & N36 & N42;
    assign N276 = N1 & N26 & N51;
    assign N279 = ~(N1 & N8 & N51 & N17);
    assign N280 = ~(N1 & N8 & N13 & N55);
    assign N284 = ~(N59 & N42 & N68 & N72);
    assign N285 = ~(N29 & N68);
    assign N286 = ~(N59 & N68 & N74);
    assign N287 = N29 & N75 & N80;
    assign N290 = N29 & N75 & N42;
    assign N291 = N29 & N36 & N80;
    assign N292 = N29 & N36 & N42;
    assign N293 = N59 & N75 & N80;
    assign N294 = N59 & N75 & N42;
    assign N295 = N59 & N36 & N80;
    assign N296 = N59 & N36 & N42;
    assign N297 = N85 & N86;
    assign N298 = N87 | N88;
    assign N301 = ~(N91 & N96);
    assign N302 = N91 | N96;
    assign N303 = ~(N101 & N106);
    assign N304 = N101 | N106;
    assign N305 = ~(N111 & N116);
    assign N306 = N111 | N116;
    assign N307 = ~(N121 & N126);
    assign N308 = N121 | N126;
    assign N309 = N8 & N138;
    assign N310 = ~N268;
    assign N316 = N51 & N138;
    assign N317 = N17 & N138;
    assign N318 = N152 & N138;
    assign N319 = ~(N59 & N156);
    assign N322 = ~(N17 | N42);
    assign N323 = N17 & N42;
    assign N324 = ~(N159 & N165);
    assign N325 = N159 | N165;
    assign N326 = ~(N171 & N177);
    assign N327 = N171 | N177;
    assign N328 = ~(N183 & N189);
    assign N329 = N183 | N189;
    assign N330 = ~(N195 & N201);
    assign N331 = N195 | N201;
    assign N332 = N210 & N91;
    assign N333 = N210 & N96;
    assign N334 = N210 & N101;
    assign N335 = N210 & N106;
    assign N336 = N210 & N111;
    assign N337 = N255 & N259;
    assign N338 = N210 & N116;
    assign N339 = N255 & N260;
    assign N340 = N210 & N121;
    assign N341 = N255 & N267;
    assign N342 = ~N269;
    assign N343 = ~N273;
    assign N344 = N270 | N273;
    assign N345 = ~N276;
    assign N346 = ~N276;
    assign N347 = ~N279;
    assign N348 = ~(N280 | N284);
    assign N349 = N280 | N285;
    assign N350 = N280 | N286;
    assign N351 = ~N293;
    assign N352 = ~N294;
    assign N353 = ~N295;
    assign N354 = ~N296;
    assign N355 = ~(N89 & N298);
    assign N356 = N90 & N298;
    assign N357 = ~(N301 & N302);
    assign N360 = ~(N303 & N304);
    assign N363 = ~(N305 & N306);
    assign N366 = ~(N307 & N308);
    assign N369 = ~N310;
    assign N375 = ~(N322 | N323);
    assign N376 = ~(N324 & N325);
    assign N379 = ~(N326 & N327);
    assign N382 = ~(N328 & N329);
    assign N385 = ~(N330 & N331);
    assign N388 = N290;
    assign N389 = N291;
    assign N390 = N292;
    assign N391 = N297;
    assign N392 = N270 | N343;
    assign N393 = ~N345;
    assign N399 = ~N346;
    assign N400 = N348 & N73;
    assign N401 = ~N349;
    assign N402 = ~N350;
    assign N403 = ~N355;
    assign N404 = ~N357;
    assign N405 = ~N360;
    assign N406 = N357 & N360;
    assign N407 = ~N363;
    assign N408 = ~N366;
    assign N409 = N363 & N366;
    assign N410 = ~(N347 & N352);
    assign N411 = ~N376;
    assign N412 = ~N379;
    assign N413 = N376 & N379;
    assign N414 = ~N382;
    assign N415 = ~N385;
    assign N416 = N382 & N385;
    assign N417 = N210 & N369;
    assign N418 = N342;
    assign N419 = N344;
    assign N420 = N351;
    assign N421 = N353;
    assign N422 = N354;
    assign N423 = N356;
    assign N424 = ~N400;
    assign N425 = N404 & N405;
    assign N426 = N407 & N408;
    assign N427 = N319 & N393 & N55;
    assign N432 = N393 & N17 & N287;
    assign N437 = ~(N393 & N287 & N55);
    assign N442 = ~(N375 & N59 & N156 & N393);
    assign N443 = ~(N393 & N319 & N17);
    assign N444 = N411 & N412;
    assign N445 = N414 & N415;
    assign N446 = N392;
    assign N447 = N399;
    assign N448 = N401;
    assign N449 = N402;
    assign N450 = N403;
    assign N451 = ~N424;
    assign N460 = ~(N406 | N425);
    assign N463 = ~(N409 | N426);
    assign N466 = ~(N442 & N410);
    assign N475 = N143 & N427;
    assign N476 = N310 & N432;
    assign N477 = N146 & N427;
    assign N478 = N310 & N432;
    assign N479 = N149 & N427;
    assign N480 = N310 & N432;
    assign N481 = N153 & N427;
    assign N482 = N310 & N432;
    assign N483 = ~(N443 & N1);
    assign N488 = N369 | N437;
    assign N489 = N369 | N437;
    assign N490 = N369 | N437;
    assign N491 = N369 | N437;
    assign N492 = ~(N413 | N444);
    assign N495 = ~(N416 | N445);
    assign N498 = ~(N130 & N460);
    assign N499 = N130 | N460;
    assign N500 = ~(N463 & N135);
    assign N501 = N463 | N135;
    assign N502 = N91 & N466;
    assign N503 = ~(N475 | N476);
    assign N504 = N96 & N466;
    assign N505 = ~(N477 | N478);
    assign N506 = N101 & N466;
    assign N507 = ~(N479 | N480);
    assign N508 = N106 & N466;
    assign N509 = ~(N481 | N482);
    assign N510 = N143 & N483;
    assign N511 = N111 & N466;
    assign N512 = N146 & N483;
    assign N513 = N116 & N466;
    assign N514 = N149 & N483;
    assign N515 = N121 & N466;
    assign N516 = N153 & N483;
    assign N517 = N126 & N466;
    assign N518 = ~(N130 & N492);
    assign N519 = N130 | N492;
    assign N520 = ~(N495 & N207);
    assign N521 = N495 | N207;
    assign N522 = N451 & N159;
    assign N523 = N451 & N165;
    assign N524 = N451 & N171;
    assign N525 = N451 & N177;
    assign N526 = N451 & N183;
    assign N527 = ~(N451 & N189);
    assign N528 = ~(N451 & N195);
    assign N529 = ~(N451 & N201);
    assign N530 = ~(N498 & N499);
    assign N533 = ~(N500 & N501);
    assign N536 = ~(N309 | N502);
    assign N537 = ~(N316 | N504);
    assign N538 = ~(N317 | N506);
    assign N539 = ~(N318 | N508);
    assign N540 = ~(N510 | N511);
    assign N541 = ~(N512 | N513);
    assign N542 = ~(N514 | N515);
    assign N543 = ~(N516 | N517);
    assign N544 = ~(N518 & N519);
    assign N547 = ~(N520 & N521);
    assign N550 = ~N530;
    assign N551 = ~N533;
    assign N552 = N530 & N533;
    assign N553 = ~(N536 & N503);
    assign N557 = ~(N537 & N505);
    assign N561 = ~(N538 & N507);
    assign N565 = ~(N539 & N509);
    assign N569 = ~(N488 & N540);
    assign N573 = ~(N489 & N541);
    assign N577 = ~(N490 & N542);
    assign N581 = ~(N491 & N543);
    assign N585 = ~N544;
    assign N586 = ~N547;
    assign N587 = N544 & N547;
    assign N588 = N550 & N551;
    assign N589 = N585 & N586;
    assign N590 = ~(N553 & N159);
    assign N593 = N553 | N159;
    assign N596 = N246 & N553;
    assign N597 = ~(N557 & N165);
    assign N600 = N557 | N165;
    assign N605 = N246 & N557;
    assign N606 = ~(N561 & N171);
    assign N609 = N561 | N171;
    assign N615 = N246 & N561;
    assign N616 = ~(N565 & N177);
    assign N619 = N565 | N177;
    assign N624 = N246 & N565;
    assign N625 = ~(N569 & N183);
    assign N628 = N569 | N183;
    assign N631 = N246 & N569;
    assign N632 = ~(N573 & N189);
    assign N635 = N573 | N189;
    assign N640 = N246 & N573;
    assign N641 = ~(N577 & N195);
    assign N644 = N577 | N195;
    assign N650 = N246 & N577;
    assign N651 = ~(N581 & N201);
    assign N654 = N581 | N201;
    assign N659 = N246 & N581;
    assign N660 = ~(N552 | N588);
    assign N661 = ~(N587 | N589);
    assign N662 = ~N590;
    assign N665 = N593 & N590;
    assign N669 = ~(N596 | N522);
    assign N670 = ~N597;
    assign N673 = N600 & N597;
    assign N677 = ~(N605 | N523);
    assign N678 = ~N606;
    assign N682 = N609 & N606;
    assign N686 = ~(N615 | N524);
    assign N687 = ~N616;
    assign N692 = N619 & N616;
    assign N696 = ~(N624 | N525);
    assign N697 = ~N625;
    assign N700 = N628 & N625;
    assign N704 = ~(N631 | N526);
    assign N705 = ~N632;
    assign N708 = N635 & N632;
    assign N712 = ~(N337 | N640);
    assign N713 = ~N641;
    assign N717 = N644 & N641;
    assign N721 = ~(N339 | N650);
    assign N722 = ~N651;
    assign N727 = N654 & N651;
    assign N731 = ~(N341 | N659);
    assign N732 = ~(N654 & N261);
    assign N733 = ~(N644 & N654 & N261);
    assign N734 = ~(N635 & N644 & N654 & N261);
    assign N735 = ~N662;
    assign N736 = N228 & N665;
    assign N737 = N237 & N662;
    assign N738 = ~N670;
    assign N739 = N228 & N673;
    assign N740 = N237 & N670;
    assign N741 = ~N678;
    assign N742 = N228 & N682;
    assign N743 = N237 & N678;
    assign N744 = ~N687;
    assign N745 = N228 & N692;
    assign N746 = N237 & N687;
    assign N747 = ~N697;
    assign N748 = N228 & N700;
    assign N749 = N237 & N697;
    assign N750 = ~N705;
    assign N751 = N228 & N708;
    assign N752 = N237 & N705;
    assign N753 = ~N713;
    assign N754 = N228 & N717;
    assign N755 = N237 & N713;
    assign N756 = ~N722;
    assign N757 = ~(N727 | N261);
    assign N758 = N727 & N261;
    assign N759 = N228 & N727;
    assign N760 = N237 & N722;
    assign N761 = ~(N644 & N722);
    assign N762 = ~(N635 & N713);
    assign N763 = ~(N635 & N644 & N722);
    assign N764 = ~(N609 & N687);
    assign N765 = ~(N600 & N678);
    assign N766 = ~(N600 & N609 & N687);
    assign N767 = N660;
    assign N768 = N661;
    assign N769 = ~(N736 | N737);
    assign N770 = ~(N739 | N740);
    assign N771 = ~(N742 | N743);
    assign N772 = ~(N745 | N746);
    assign N773 = ~(N750 & N762 & N763 & N734);
    assign N777 = ~(N748 | N749);
    assign N778 = ~(N753 & N761 & N733);
    assign N781 = ~(N751 | N752);
    assign N782 = ~(N756 & N732);
    assign N785 = ~(N754 | N755);
    assign N786 = ~(N757 | N758);
    assign N787 = ~(N759 | N760);
    assign N788 = ~(N700 | N773);
    assign N789 = N700 & N773;
    assign N790 = ~(N708 | N778);
    assign N791 = N708 & N778;
    assign N792 = ~(N717 | N782);
    assign N793 = N717 & N782;
    assign N794 = N219 & N786;
    assign N795 = ~(N628 & N773);
    assign N796 = ~(N795 & N747);
    assign N802 = ~(N788 | N789);
    assign N803 = ~(N790 | N791);
    assign N804 = ~(N792 | N793);
    assign N805 = ~(N340 | N794);
    assign N806 = ~(N692 | N796);
    assign N807 = N692 & N796;
    assign N808 = N219 & N802;
    assign N809 = N219 & N803;
    assign N810 = N219 & N804;
    assign N811 = ~(N805 & N787 & N731 & N529);
    assign N812 = ~(N619 & N796);
    assign N813 = ~(N609 & N619 & N796);
    assign N814 = ~(N600 & N609 & N619 & N796);
    assign N815 = ~(N738 & N765 & N766 & N814);
    assign N819 = ~(N741 & N764 & N813);
    assign N822 = ~(N744 & N812);
    assign N825 = ~(N806 | N807);
    assign N826 = ~(N335 | N808);
    assign N827 = ~(N336 | N809);
    assign N828 = ~(N338 | N810);
    assign N829 = ~N811;
    assign N830 = ~(N665 | N815);
    assign N831 = N665 & N815;
    assign N832 = ~(N673 | N819);
    assign N833 = N673 & N819;
    assign N834 = ~(N682 | N822);
    assign N835 = N682 & N822;
    assign N836 = N219 & N825;
    assign N837 = ~(N826 & N777 & N704);
    assign N838 = ~(N827 & N781 & N712 & N527);
    assign N839 = ~(N828 & N785 & N721 & N528);
    assign N840 = ~N829;
    assign N841 = ~(N815 & N593);
    assign N842 = ~(N830 | N831);
    assign N843 = ~(N832 | N833);
    assign N844 = ~(N834 | N835);
    assign N845 = ~(N334 | N836);
    assign N846 = ~N837;
    assign N847 = ~N838;
    assign N848 = ~N839;
    assign N849 = N735 & N841;
    assign N850 = N840;
    assign N851 = N219 & N842;
    assign N852 = N219 & N843;
    assign N853 = N219 & N844;
    assign N854 = ~(N845 & N772 & N696);
    assign N855 = ~N846;
    assign N856 = ~N847;
    assign N857 = ~N848;
    assign N858 = ~N849;
    assign N859 = ~(N417 | N851);
    assign N860 = ~(N332 | N852);
    assign N861 = ~(N333 | N853);
    assign N862 = ~N854;
    assign N863 = N855;
    assign N864 = N856;
    assign N865 = N857;
    assign N866 = N858;
    assign N867 = ~(N859 & N769 & N669);
    assign N868 = ~(N860 & N770 & N677);
    assign N869 = ~(N861 & N771 & N686);
    assign N870 = ~N862;
    assign N871 = ~N867;
    assign N872 = ~N868;
    assign N873 = ~N869;
    assign N874 = N870;
    assign N875 = ~N871;
    assign N876 = ~N872;
    assign N877 = ~N873;
    assign N878 = N875;
    assign N879 = N876;
    assign N880 = N877;

endmodule
"""
# --OPTION--

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    exp_dir = p(args.save_dir).resolve() if args.save_dir else p(create_save_dir("trained")).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    design_path = pj(exp_dir, "design.v")
    with open(design_path, "w") as f:
        f.write(generate_seed_verilog().strip())

    print(f"Seed C880 generated at: {design_path}")

if __name__ == "__main__":
    main()
