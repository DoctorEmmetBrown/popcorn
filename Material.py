import sys
import xraydb
import numpy as np


class Material:
    def __init__(self, name):
        """Material class constructor.

        :param name: material's name
        :type name: string

        Notes:
            If the material doesn't correspond to any existing material, the user will be asked to create one.
        """
        self.name = name
        # If the material exists, we retrieve its formula and density
        if xraydb.find_material(name):
            self.chemical_formula, self.density = xraydb.get_material(name)
        # If the material doesn't exist, we ask the user to create a new one
        else:
            # Asking the user if he wants to create a new material
            user_material_creation = input(
                "Material " + str(name) + " doesn't seem to exist, do you want to create it ? [Y/N]: ")
            if "y" in user_material_creation.lower():
                # We ask the user to give the chemical formula with the proper format
                user_chemical_formula = input(
                    "Please enter the chemical formula for your new material.\n >Example: Au0.55Xe0.35Gd0.1\n")
                while not xraydb.validate_formula(user_chemical_formula):
                    user_chemical_formula = input("Chemical formula not valid.\n >Example: Au0.55Xe0.35Gd0.1\n")
                # Once the given formula is valid, we ask the user to give the materials density
                user_density = float(input("Please enter the material's density: "))

                # We add the material to the materials list
                xraydb.add_material(name, user_chemical_formula, user_density)

                # We initialize the formula and density attributes
                self.chemical_formula = user_chemical_formula
                self.density = user_density

            # If the user doesn't want to create a new material
            else:
                sys.exit("Please try again with an existing material formula.")

    def get_mu(self, energy, density=None):
        """linear attenuation of the material at given energies

        :param energy: energy or array of energies in eV
        :type energy: float | numpy.ndarray
        :param density: material density (g/cm^3)
        :type density: None or float
        :return: absorption length in 1/cm
        :rtype: float | numpy.ndarray
        """
        return xraydb.material_mu(self.chemical_formula, energy, density)

    def get_beta(self, energy, density):
        """beta coefficient of the material of given density at given energy

        :param energy: energy in eV
        :type energy: float
        :param density: material density (g/cm^3)
        :type density: float
        :return: beta coefficient
        :rtype: float
        """
        return xraydb.xray_delta_beta(self.chemical_formula, density, energy)[1]

    def get_delta(self, energy, density):
        """delta coefficient of the material of given density at given energy

        :param energy: energy in eV
        :type energy: float
        :param density: material density (g/cm^3)
        :type density: float
        :return: delta coefficient
        :rtype: float
        """
        return xraydb.xray_delta_beta(self.chemical_formula, density, energy)[0]

    def get_beta_from_spectrum(self, energy_spectrum, density):
        """beta coefficient of the material of given density at given energies

        :param energy_spectrum: array of energies in eV
        :type energy_spectrum: numpy.ndarray
        :param density: material density (g/cm^3)
        :type density: float
        :return: beta coefficients for each given energy
        :rtype: float
        """
        list_of_betas = []
        for energy in energy_spectrum:
            list_of_betas.append(self.get_beta(int(energy), density))
        return list_of_betas

    def get_delta_from_spectrum(self, energy_spectrum, density):
        """delta coefficient of the material of given density at given energies

        :param energy_spectrum: array of energies in eV
        :type energy_spectrum: numpy.ndarray
        :param density: material density (g/cm^3)
        :type density: float
        :return: delta coefficients for each given energy
        :rtype: float
        """
        list_of_deltas = []
        for energy in energy_spectrum:
            list_of_deltas.append(self.get_delta(int(energy), density))
        return list_of_deltas


if __name__ == '__main__':
    gold = Material("Gold")
    energies = np.array([10000, 35000, 40000, 45000, 50000])
    print("Mus :", gold.get_mu(energies))
    print("Beta :", gold.get_beta(int(energies[0]), 19.3))
    print("Betas :", gold.get_beta_from_spectrum(energies, 19.3))
    print("Delta :", gold.get_delta(int(energies[0]), 19.3))
    print("Deltas :", gold.get_delta_from_spectrum(energies, 19.3))
