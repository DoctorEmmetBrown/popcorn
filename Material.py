import sys
import xraydb
import numpy as np


class Material:
    def __init__(self, name):
        """
        Material class constructor
            Args:
                name (string):   material's name

            Returns:
                None

            Notes:
                If the material doesn't correspond to any existing material, the user will be asked to create one.

            Examples:
                >>> gold = Material("Gold")
            """
        self.name = name
        if xraydb.find_material(name):
            self.chemicalFormula, self.density = xraydb.get_material(name)
        else:
            userMaterialCreation = input("Material " + str(name) + " doesn't seem to exist, do you want to create it ? [Y/N]: ")
            if "y" in userMaterialCreation.lower():
                userChemicalFormula = input("Please enter the chemical formula for your new material.\n >Example: Au0.55Xe0.35Gd0.1\n")
                while not xraydb.validate_formula(userChemicalFormula):
                    userChemicalFormula = input("Chemical formula not valid.\n >Example: Au0.55Xe0.35Gd0.1\n")
                userDensity = float(input("Please enter the material's density: "))
                xraydb.add_material(name, userChemicalFormula, userDensity)
                self.chemicalFormula = userChemicalFormula
                self.density = userDensity
            else:
                sys.exit("Please try again with an existing material formula.")


    def getMu(self, energies, density=None):
        """Linear attenuation of the material at given energies

            Args:
                energies (float or ndarray): energy or array of energies in eV
                density (None or float):   material density (g/cm^3).
            Returns:
                Absorption length in 1/cm
            Notes:
                If density is None and material is known, that density will be used.

            Examples:
                >>> gold.getMu(10000.0)
                2279.92920598
            """
        return xraydb.material_mu(self.chemicalFormula, energies, density)


    def getBeta(self, energy, density):
        """
        Beta coefficient of the material of given density at given energy
            Args:
                energy (float): energy in eV
                density (float): material density (g/cm^3).

            Returns:
                Beta coefficient

            Examples:
                >>> gold.getBeta(10000.0, 19.3)
                2.084575845737037e-06
            """
        return xraydb.xray_delta_beta(self.chemicalFormula, density, energy)[1]


    def getDelta(self, energy, density):
        """
        Beta coefficient of the material of given density at given energy
            Args:
                energy (float): energy in eV
                density (float): material density (g/cm^3).

            Returns:
                Beta coefficient

            Examples:
                >>> gold.getBeta(10000.0, 19.3)
                2.084575845737037e-06
            """
        return xraydb.xray_delta_beta(self.chemicalFormula, density, energy)[0]


    def getBetaFromSpectrum(self, energySpectrum, density):
        """
        Beta coefficient of the material of given density at given energies
            Args:
                energySpectrum (ndarray of float): array of energies in eV
                density (float): material density (g/cm^3).

            Returns:
                Beta coefficients for each given energy

            Examples:
                >>> gold.getBetaFromSpectrum(np.array([10000, 35000, 40000, 45000, 50000]), 19.3)
                [2.084575845737037e-06, 9.090995289307478e-08, 5.519117987481352e-08, 3.5635111431158425e-08, 2.4144113225644783e-08]
            """
        listOfBeta = []
        for energy in energySpectrum:
            listOfBeta.append(self.getBeta(int(energy), density))
        return listOfBeta


    def getDeltaFromSpectrum(self, energySpectrum, density):
        """
        Delta coefficient of the material of given density at given energies
            Args:
                energySpectrum (ndarray of float): array of energies in eV
                density (float): material density (g/cm^3).

            Returns:
                Delta coefficients for each given energy

            Examples:
                >>> gold.getDeltaFromSpectrum(np.array([10000, 35000, 40000, 45000, 50000]), 19.3)
                [2.970837201294633e-05, 2.5979065733569894e-06, 1.9844421586969203e-06, 1.564375753386381e-06, 1.2643572704001252e-06]
            """
        listOfDelta = []
        for energy in energySpectrum:
            listOfDelta.append(self.getDelta(int(energy), density))
        return listOfDelta


if __name__ == '__main__':
    gold = Material("Gold")
    energies = np.array([10000, 35000, 40000, 45000, 50000])
    print("Mus :", gold.getMu(energies))
    print("Beta :", gold.getBeta(int(energies[0]), 19.3))
    print("Betas :", gold.getBetaFromSpectrum(energies, 19.3))
    print("Delta :", gold.getDelta(int(energies[0]), 19.3))
    print("Deltas :", gold.getDeltaFromSpectrum(energies, 19.3))